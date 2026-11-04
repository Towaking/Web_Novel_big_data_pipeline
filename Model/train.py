"""
Entry point pelatihan model
Skrip mandiri untuk melatih ulang model tanpa menjalankan seluruh pipeline.
Berguna saat yang berubah hanya arsitektur atau hyperparameter, sedangkan
data bersih di lake sudah benar.

Skrip ini TIDAK menggandakan logika training — ia memanggil modul Fase 4 yang
sama (`src/phases/phase4_model.py`, `genre_models.py`).
Menyalin kode training ke sini akan melahirkan dua kebenaran yang perlahan
menyimpang; yang ada di sini hanyalah lapisan antarmuka baris perintah.

Contoh pemakaian
    # Latih ulang semuanya (3 model genre klasik + ensemble + rekomendasi + klaster + rating)
    docker compose exec pipeline python Model/train.py

    # Hanya model genre bawaan (logreg, svm, rf, ensemble)
    docker compose exec pipeline python Model/train.py --only genre

    # Hanya arsitektur tertentu (yang lain dipertahankan dari artefak lama)
    docker compose exec pipeline python Model/train.py --models svm rf ensemble

    # Latih tanpa mengunggah artefak ke MinIO
    docker compose exec pipeline python Model/train.py --no-upload

Daftar arsitektur yang tersedia:
    logreg   Logistic Regression (TF-IDF)
    svm      Linear SVM terkalibrasi (TF-IDF)
    rf       Random Forest multi-output (TF-IDF terseleksi chi-kuadrat)
    ensemble rata-rata berbobot svm + rf (butuh keduanya di run yang sama)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Skrip ini berada di dalam Model/, jadi akar proyek perlu ditambahkan ke
# jalur impor supaya `python Model/train.py` bekerja tanpa PYTHONPATH khusus.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402

from src.common.logging_utils import get_logger, log_section  # noqa: E402
from src.phases import genre_models, phase4_model  # noqa: E402

logger = get_logger("model.train")
ARTIFACT_DIR = phase4_model.ARTIFACT_DIR


def train_genre_only(model_names: list[str] | None, upload: bool) -> dict:
    """
    Latih ulang hanya Head A (pelabelan genre).

    Model yang tidak diminta tetap dipertahankan dari artefak sebelumnya,
    sehingga melatih ulang satu arsitektur tidak menghapus yang lain dari
    dashboard.
    """
    log_section(logger, "melatih ulang model genre")

    requested = model_names or list(genre_models.DEFAULT_MODELS)
    unknown = [m for m in requested if m not in genre_models.MODEL_CATALOG]
    if unknown:
        raise SystemExit(
            f"Arsitektur tidak dikenal: {', '.join(unknown)}. "
            f"Pilihan: {', '.join(genre_models.MODEL_CATALOG)}"
        )

    pdf, stopwords = phase4_model.load_training_frame()
    frame = phase4_model.build_genre_frame(pdf, stopwords)

    artifacts, mlb, metrics = phase4_model.train_genre_tagger(pdf, names=requested, frame=frame)

    previous_path = ARTIFACT_DIR / "genre_models.joblib"
    if previous_path.is_file() and set(requested) != set(genre_models.DEFAULT_MODELS):
        # Model lama dipertahankan hanya bila ruang labelnya sama dengan binarizer baru; genre bisa bertambah sejak model itu dilatih.
        old_binarizer = ARTIFACT_DIR / "genre_binarizer.joblib"
        old_labels = list(joblib.load(old_binarizer).classes_) if old_binarizer.is_file() else None
        if old_labels != list(mlb.classes_):
            raise SystemExit(
                f"Ruang label berubah ({len(old_labels or [])} -> {len(mlb.classes_)} genre) sejak "
                "genre_models.joblib terakhir dilatih: model lama tidak bisa digabung. "
                "Latih ulang model bawaan (tanpa --models)."
            )
        previous = joblib.load(previous_path)
        merged = {**previous, **artifacts}
        ensemble = merged.get("ensemble", {})
        if "ensemble" not in artifacts and any(m in artifacts for m in ensemble.get("members", {})):
            # Bobot & ambang ensemble disetel untuk anggota lama.
            merged.pop("ensemble")
            logger.info("Ensemble lama dibuang karena anggotanya dilatih ulang; sertakan 'ensemble' "
                        "bersama anggotanya di --models untuk merakitnya lagi.")
        logger.info(
            "Artefak digabung: %s dilatih ulang, %s dipertahankan.",
            ", ".join(requested),
            ", ".join(k for k in previous if k not in requested) or "(tidak ada)",
        )
    else:
        merged = artifacts

    metrics_path = ARTIFACT_DIR / "metrics.json"
    all_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    head_a = all_metrics.get("head_a_genre_tagger", {})
    head_a.setdefault("models", {})
    head_a["models"].update(metrics["models"])
    head_a.update({k: v for k, v in metrics.items() if k != "models"})

    # Buang metrik arsitektur yang artefaknya sudah tidak ada, agar tabel perbandingan dashboard tidak memuat model yang tak bisa dipilih.
    stale = [name for name in head_a["models"] if name not in merged]
    for name in stale:
        head_a["models"].pop(name)
    if stale:
        logger.info("Metrik arsitektur usang dibuang: %s", ", ".join(stale))

    # Pilihan dari skor OOF (set dev), bukan skor uji - sama seperti train_all.
    head_a["best_model"] = max(
        head_a["models"],
        key=lambda k: head_a["models"][k].get("f1_macro_oof", head_a["models"][k]["f1_macro"]),
    )
    all_metrics["head_a_genre_tagger"] = head_a
    all_metrics.setdefault("stopwords", stopwords)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    phase4_model.save_artifacts(genre_models=merged, genre_binarizer=mlb)
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    logger.info("Metrik diperbarui: %s", metrics_path)

    if upload:
        phase4_model._upload_artifacts_to_lake()

    _print_comparison(head_a["models"], head_a["best_model"])
    return all_metrics


def _print_comparison(models: dict, best: str) -> None:
    """Tabel perbandingan antar arsitektur, dicetak ke log."""
    log_section(logger, "perbandingan model genre (set uji)")
    header = f"{'model':<22} {'F1 makro':>9} {'F1 mikro':>9} {'P@3':>7} {'Hamming':>8}"
    logger.info(header)
    logger.info("-" * len(header))
    for name, entry in sorted(models.items(), key=lambda kv: -kv[1]["f1_macro"]):
        mark = " *" if name == best else "  "
        logger.info(
            "%-22s %9.4f %9.4f %7.4f %8.4f%s",
            entry.get("label", name),
            entry["f1_macro"],
            entry["f1_micro"],
            entry.get("precision_at_3", float("nan")),
            entry.get("hamming_loss", float("nan")),
            mark,
        )
    logger.info("(*) model default yang dipakai dashboard bila pengguna tidak memilih")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Latih ulang model NovelLens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--only",
        choices=["all", "genre"],
        default="all",
        help="'genre' hanya melatih Head A; 'all' melatih seluruh head (default).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="NAMA",
        help="Arsitektur genre yang dilatih ulang (default: logreg svm rf ensemble). "
        "Menyiratkan --only genre.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Jangan menyalin artefak ke MinIO setelah pelatihan.",
    )
    args = parser.parse_args()

    try:
        if args.models or args.only == "genre":
            train_genre_only(args.models, upload=not args.no_upload)
        else:
            metrics = phase4_model.run()
            _print_comparison(
                metrics["head_a_genre_tagger"]["models"],
                metrics["head_a_genre_tagger"]["best_model"],
            )
    except KeyboardInterrupt:
        logger.warning("Pelatihan dihentikan pengguna. Artefak lama tidak diubah.")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.error("PELATIHAN GAGAL: %s", exc, exc_info=True)
        return 1

    logger.info("Selesai. Restart container Streamlit agar artefak baru terbaca:")
    logger.info("    docker compose restart streamlit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
