"""
Orchestrator pipeline
Menjalankan Fase 1 → 4 secara berurutan, lalu langkah Big Data (variety) dan
lapisan tata kelola data (inventaris elemen → model data → audit), dalam satu
proses.

Perilaku saat gagal: pipeline BERHENTI di langkah pertama yang gagal.
Melanjutkan ke langkah berikutnya di atas data yang tidak lengkap akan
menghasilkan analitik dan model yang "sukses" tetapi salah — kegagalan diam-
diam yang jauh lebih berbahaya daripada berhenti lebih awal.

Bukan Airflow/Dagster - ini penjadwal SEKUENSIAL sederhana. Yang disediakan:
  * `--retry N`   — ulangi satu langkah yang gagal hingga N kali (jeda 5 detik)
                    sebelum benar-benar menyerah. Berguna untuk kegagalan
                    transien (mis. NameNode masih safe mode, jaringan MinIO).
  * `--resume`    — lanjutkan dari langkah yang gagal terakhir, bukan dari
                    awal. Status disimpan di `logs/pipeline_state.json` per
                    langkah yang SUKSES; file itu otomatis terhapus begitu
                    seluruh invocation berhasil (tidak ada yang perlu di-resume
                    lagi). Hanya berlaku bila argumen run (`--from/--to/--only/
                    governance`) SAMA dengan invocation yang gagal - beda
                    argumen berarti run baru, status lama diabaikan.

Jalankan:
    docker compose exec pipeline python scripts/run_pipeline.py
    docker compose exec pipeline python scripts/run_pipeline.py --from 3
    docker compose exec pipeline python scripts/run_pipeline.py --only 2
    docker compose exec pipeline python scripts/run_pipeline.py --only-governance
    docker compose exec pipeline python scripts/run_pipeline.py --retry 2
    docker compose exec pipeline python scripts/run_pipeline.py --resume   # setelah run sebelumnya gagal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.common.config import get_settings
from src.common.logging_utils import get_logger, log_section
from src.common.spark_session import restart_shared_session, shared_spark_session

logger = get_logger("orchestrator")

STATE_PATH = Path("logs/pipeline_state.json")
RETRY_DELAY_SECONDS = 5


def _phase1(force: bool):
    from src.phases import phase1_collect

    return phase1_collect.run(force=force)


def _phase2(_force: bool):
    from src.phases import phase2_clean

    return phase2_clean.run()


def _phase3(_force: bool):
    from src.phases import phase3_explore

    return phase3_explore.run()


def _phase4(_force: bool):
    from src.phases import phase4_model

    return phase4_model.run()


def _catalog(_force: bool):
    from src.phases import data_catalog

    return data_catalog.run()


def _datamodel(_force: bool):
    from src.phases import data_model

    return data_model.run()


def _graphanalysis(_force: bool):
    from src.phases import graph_analysis

    return graph_analysis.run()


def _audit(_force: bool):
    from src.phases import governance_audit

    return governance_audit.run()


def _variety(_force: bool):
    from src.phases import data_variety

    return data_variety.run()


PHASES = {
    1: ("Collect (Ingestion ke MinIO)", _phase1),
    2: ("Clean (PySpark preprocessing)", _phase2),
    3: ("Explore (Spark SQL analytics)", _phase3),
    4: ("Model (Arsitektur AI multi-head)", _phase4),
}

#: Langkah Big Data yang memakai hasil Fase 2 tetapi bukan fase bernomor. HDFS dan MapReduce dijalankan
#: dari container Hadoop (`make hdfs-load mapreduce`).
EXPORT_STEPS = (
    ("variety", "Pemisahan dataset per format (CSV/JSONL/TXT)", _variety),
)

#: Lapisan tata kelola data: memotong seluruh fase, jadi tidak diberi nomor fase.
#: Dijalankan setelah fase numerik karena bergantung pada zona raw dan clean yang sudah terisi.
GOVERNANCE_STEPS = (
    ("catalog", "Inventaris elemen data & format", _catalog),
    ("datamodel", "Model data (star schema + graf)", _datamodel),
    ("graphanalysis", "Analisis graf (PageRank, komunitas, traversal)", _graphanalysis),
    ("audit", "Audit tata kelola IT", _audit),
)


def _state_path() -> Path:
    return get_settings().paths.project_root / STATE_PATH


def _run_signature(start: int, end: int, phases: bool, governance: bool) -> str:
    """Identitas satu 'jenis run' - resume hanya sah untuk run yang SAMA."""
    return f"{start}-{end}-{phases}-{governance}"


def _load_completed(signature: str) -> set[str]:
    path = _state_path()
    if not path.is_file():
        return set()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return set()
    if state.get("signature") != signature:
        logger.info("Status resume lama tidak cocok dengan run ini (argumen berbeda) - diabaikan.")
        return set()
    return set(state.get("completed", []))


def _save_completed(signature: str, completed: list[str]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"signature": signature, "completed": completed}, indent=2), encoding="utf-8")


def _clear_state() -> None:
    path = _state_path()
    if path.is_file():
        path.unlink()


def _warn_stale_models() -> None:
    """
    Zona clean baru saja ditulis ulang tanpa Fase 4: artefak model (AI Studio,
    halaman Methodology) masih dari zona clean versi lama. Tidak menggagalkan
    run, tetapi jangan biarkan basinya diam-diam.
    """
    metrics_path = get_settings().paths.model_dir / "artifacts" / "metrics.json"
    if not metrics_path.is_file():
        return
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return
    logger.warning(
        "Artefak model Fase 4 TIDAK diperbarui pada run ini (dilatih %s atas %s dokumen). "
        "Zona clean sudah berubah - jalankan `--only 4` agar dashboard AI Studio memakai data yang sama.",
        metrics.get("generated_at_utc", "?"), f"{metrics.get('n_documents', 0):,}",
    )


def _run_step(key: str, label: str, func, force: bool, durations: list, retries: int,
              results: dict) -> bool:
    """
    Jalankan satu langkah, dengan retry opsional untuk kegagalan transien.

    Nilai balik langkah disimpan di `results[key]`. Kembalikan False hanya
    setelah SELURUH percobaan (1 + retries) gagal.
    """
    attempt = 0
    while True:
        attempt += 1
        started = time.perf_counter()
        try:
            results[key] = func(force)
        except Exception as exc:  # noqa: BLE001 - berhenti/ulangi dengan pesan yang jelas
            elapsed = time.perf_counter() - started
            # Sesi Spark bersama bisa ikut rusak (executor hilang, konteks mati):
            # percobaan ulang dan langkah sesudahnya mulai dari sesi baru.
            restart_shared_session()
            if attempt <= retries:
                logger.warning(
                    "%s GAGAL (percobaan %d/%d, %.1f detik): %s - mengulang dalam %ds",
                    label, attempt, retries + 1, elapsed, exc, RETRY_DELAY_SECONDS,
                )
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            logger.error("%s GAGAL setelah %d percobaan (%.1f detik terakhir): %s",
                         label, attempt, elapsed, exc, exc_info=True)
            return False
        durations.append((label, time.perf_counter() - started))
        return True


def run_pipeline(
    start: int = 1,
    end: int = 4,
    force: bool = False,
    governance: bool = True,
    phases: bool = True,
    retries: int = 0,
    resume: bool = False,
) -> int:
    log_section(logger, "menjalankan pipeline big data novel")
    # Satu aplikasi Spark untuk seluruh langkah invocation ini, bukan satu per
    # langkah: tiap aplikasi baru meluncurkan ulang executor di worker.
    with shared_spark_session("novel-pipeline"):
        return _run_pipeline(start, end, force, governance, phases, retries, resume)


def _run_pipeline(start: int, end: int, force: bool, governance: bool, phases: bool,
                  retries: int, resume: bool) -> int:
    # Variety hanya berguna dijalankan bila Fase 2 BENAR-BENAR dieksekusi ulang
    # pada invocation ini (zona clean baru saja ditulis) - bukan sekadar
    # `end >= 2`, yang salah menjalankannya juga untuk `--only 4` (start=4).
    run_export = phases and start <= 2 <= end
    if phases and not run_export:
        logger.info("Fase 2 tidak ikut dijalankan: langkah pemisahan format dilewati.")
    durations: list[tuple[str, float]] = []

    signature = _run_signature(start, end, phases, governance)
    done = _load_completed(signature) if resume else set()
    if done:
        logger.info("Resume aktif - langkah yang sudah sukses sebelumnya akan dilewati: %s", sorted(done))
    completed = list(done)
    results: dict = {}

    def step(key: str, label: str, func) -> bool:
        if key in done:
            logger.info("### %s — DILEWATI (sudah sukses pada percobaan sebelumnya)", label)
            return True
        if not _run_step(key, label, func, force, durations, retries, results):
            return False
        completed.append(key)
        _save_completed(signature, completed)
        return True

    if phases:
        for number in range(start, end + 1):
            label, func = PHASES[number]
            logger.info("")
            if not step(f"phase{number}", f"FASE {number} — {label}", func):
                logger.error("Pipeline dihentikan. Langkah berikutnya TIDAK dijalankan. "
                             "Perbaiki masalahnya lalu jalankan ulang dengan --resume.")
                return 1

    if run_export:
        for name, label, func in EXPORT_STEPS:
            logger.info("")
            if not step(name, f"BIG DATA — {label}", func):
                logger.error("Langkah pemisahan format dihentikan.")
                return 1

    if governance:
        for name, label, func in GOVERNANCE_STEPS:
            logger.info("")
            if not step(name, f"TATA KELOLA — {label}", func):
                logger.error("Lapisan tata kelola dihentikan.")
                return 1

    _clear_state()

    log_section(logger, "ringkasan eksekusi")
    for label, elapsed in durations:
        logger.info("%-52s | %6.1f detik", label, elapsed)
    logger.info("Total: %.1f detik", sum(d for _, d in durations))
    logger.info("Dashboard: http://localhost:8501")
    if phases and start <= 2 <= end < 4:
        _warn_stale_models()

    # Audit mengembalikan tally, bukan melempar exception: kontrol GAGAL tidak menghentikan langkah,
    # tetapi exit code mengikuti `governance_audit.main()` (2).
    tally = results.get("audit") or {}
    n_failed = tally.get("GAGAL", 0) if isinstance(tally, dict) else 0
    if n_failed:
        logger.error("Audit tata kelola: %d kontrol GAGAL - lihat Doc/generated/governance_audit.md", n_failed)
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrator pipeline Fase 1-4")
    parser.add_argument("--from", dest="start", type=int, default=1, choices=sorted(PHASES),
                        help="Fase awal (default 1)")
    parser.add_argument("--to", dest="end", type=int, default=4, choices=sorted(PHASES),
                        help="Fase akhir (default 4)")
    parser.add_argument("--only", type=int, choices=sorted(PHASES),
                        help="Jalankan satu fase saja")
    parser.add_argument("--force", action="store_true",
                        help="Paksa ingestion ulang di Fase 1 walau checksum sama")
    parser.add_argument("--skip-governance", action="store_true",
                        help="Lewati lapisan tata kelola (inventaris, model data, audit)")
    parser.add_argument("--only-governance", action="store_true",
                        help="Jalankan HANYA lapisan tata kelola, tanpa Fase 1-4")
    parser.add_argument("--retry", type=int, default=0, metavar="N",
                        help="Ulangi langkah yang gagal hingga N kali (jeda 5 detik) sebelum menyerah")
    parser.add_argument("--resume", action="store_true",
                        help="Lanjutkan dari langkah gagal terakhir (butuh argumen run yang SAMA)")
    args = parser.parse_args()

    if args.skip_governance and args.only_governance:
        parser.error("--skip-governance dan --only-governance saling bertentangan")
    if args.retry < 0:
        parser.error("--retry tidak boleh negatif")

    start, end = (args.only, args.only) if args.only else (args.start, args.end)
    if start > end:
        parser.error("--from tidak boleh lebih besar dari --to")
    return run_pipeline(
        start=start,
        end=end,
        force=args.force,
        governance=not args.skip_governance,
        phases=not args.only_governance,
        retries=args.retry,
        resume=args.resume,
    )


if __name__ == "__main__":
    sys.exit(main())
