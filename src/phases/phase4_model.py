"""
FASE 4 — MODEL (Autonomous AI Architecture)

PEMILIHAN ARSITEKTUR — alasan logis
Dataset ini punya tiga sifat yang menentukan pilihan model:

1. **Tidak ada interaksi user-item.** Tidak ada `user_id` maupun histori baca,
   jadi *collaborative filtering* (ALS) mustahil dilatih. Sistem rekomendasi
   harus berbasis konten.
2. **Sinyal teksnya campuran prosa + label.** Korpus = `title` + `synopsis` +
   `tags`: sinopsis (median 118 kata, cakupan 98,45%) memberi konteks naratif,
   sementara tag memberi label semantik yang sudah ternormalisasi. TF-IDF
   menangkap kekhasan frasa seperti "Time Loop" atau "Cultivation" dengan
   sangat baik pada skala 4.4k dokumen; yang TIDAK ditangkapnya adalah
   parafrasa — dua sinopsis bercerita sama dengan kata berbeda tidak berbagi
   satu token pun. Novel tanpa sinopsis tetap punya sinyal dari tag, jadi tidak
   ada yang gugur.

Karena itu arsitekturnya adalah **satu representasi bersama, empat kepala model**
(*shared representation, multi-head*):

        corpus_clean (title + synopsis + tags)
                 │
          TF-IDF (1-2 gram)            ← ruang sparse
                 │
        Truncated SVD (LSA, 200 dim)   ← ruang padat, tahan sinonim
                 │
   ┌─────────────┼──────────────┬──────────────────┐
   │             │              │                  │
 HEAD A        HEAD B         HEAD C            HEAD D
 Genre         Content-based  Archetype         Rating
 Tagger        Recommender    Clustering        Estimator
   │           (cosine kNN)   (KMeans+silh.)    (HistGradientBoosting)
   ├── Logistic Regression  (linear, TF-IDF sparse)
   ├── Linear SVM           (margin, kalibrasi Platt)
   ├── Random Forest        (pohon multi-output, TF-IDF terseleksi chi-kuadrat)
   └── Ensemble SVM + RF    (rata-rata probabilitas berbobot)

Head A melatih tiga model ML klasik dan satu ensemble yang dapat dipilih pengguna
di dashboard; semuanya diuji pada pembagian data yang sama sehingga perbandingannya
adil, dan model bawaan dipilih dari skor OOF.
Detail arsitektur & penyetelan ambangnya ada di `src/phases/genre_models.py`.

Head B/C/D memakai representasi bersama di atas: vectorizer hanya dilatih
sekali, artefaknya kecil, dan ketiganya dijamin "melihat" teks dengan cara
yang sama — jika ada yang perlu diperbaiki, cukup satu titik yang diubah.

Peran tiap kepala terhadap pertanyaan bisnis:
  * HEAD A — melengkapi metadata katalog yang genre-nya kosong/salah label.
  * HEAD B — "pembaca yang suka X sebaiknya membaca apa" tanpa data perilaku.
  * HEAD C — segmentasi katalog menjadi arketipe cerita untuk perencanaan konten.
  * HEAD D — estimasi penerimaan pembaca sebelum sebuah judul punya rating.

Jalankan:
    docker compose exec pipeline python -m src.phases.phase4_model
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import functions as F
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from Model.pipeline.rating import HGB_PARAMS, RatingModel
from src.common.artifact_io import dump_artifact
from src.common.config import get_settings
from src.common.logging_utils import get_logger, log_section
from src.common.spark_session import spark_session
from src.phases import genre_models
from src.phases.phase2_clean import DOMAIN_STOPWORDS

logger = get_logger("phase4.model")

SVD_COMPONENTS = 200        # ~menjelaskan mayoritas variansi pada korpus sependek ini
MIN_GENRE_SUPPORT = 50      # genre dengan <50 novel tidak cukup untuk dilatih
KMEANS_K_GRID = (5, 6, 7, 8, 9, 10)
RANDOM_STATE = 42
TEST_SIZE = 0.2

ARTIFACT_DIR = Path("/opt/project/Model/artifacts")


def load_training_frame() -> tuple[pd.DataFrame, list[str]]:
    """
    Tarik data bersih dari lake lewat Spark, lalu materialisasi ke pandas.

    Sesuai aturan optimasi: seluruh penyaringan dan pemilihan kolom dilakukan
    terdistribusi di Spark; yang di-`collect` hanya kolom yang benar-benar
    dipakai model (39.660 baris x 26 kolom, ~60 MB) — cukup kecil untuk memori
    driver dan jauh lebih cepat dilatih di sklearn daripada di Spark ML
    (perbandingannya di Fase 5, `phase5_mllib.py`).
    """
    settings = get_settings()
    clean_uri = settings.minio.s3a_uri(settings.minio.bucket_clean, "novels_clean")

    with spark_session("phase4-model") as spark:
        df = spark.read.parquet(clean_uri)
        selected = df.filter(df.has_text_signal).select(
            "series_id", "title", "url", "corpus_clean", "genre_list", "primary_genre",
            "type_origin", "type", "year", "chapter_count", "n_tags", "token_count",
            "is_completed", "rating_score", "rating_votes", "weighted_rating",
            "is_rated_reliable", "reading_list_count", "tags", "synopsis_clean",
            # Sinopsis mentah: korpus dibangun ulang darinya dengan kata bersensor dipulihkan.
            "synopsis",
            # Sinopsis dipotong 400 karakter: cukup untuk cuplikan di dashboard,
            # tanpa membuat artefak katalog membengkak beberapa megabyte.
            F.substring(F.col("synopsis"), 1, 400).alias("synopsis_snippet"),
            "synopsis_word_count", "has_synopsis",
            # Kecepatan rilis (hari per bab) — fitur numerik baru untuk Head D.
            "days_per_chapter",
            # Bukan fitur model. Baris tier 'listing' ber-series_id sintetis
            # negatif, jadi setelah diurutkan selalu berada paling atas dengan
            # bab/rating NULL. Kolom ini membuat alasannya terlihat.
            "metadata_tier",
        )
        pdf = selected.toPandas()

        # Daftar stopword yang PERSIS dipakai Fase 2 ikut disimpan sebagai
        # artefak, supaya jalur inferensi tidak pernah menyimpang dari training.
        stopwords = list(StopWordsRemover.loadDefaultStopWords("english")) + list(DOMAIN_STOPWORDS)

    # Urutan baris dikunci dengan `series_id`: `split_indices` membagi berdasarkan posisi baris, dan urutan `toPandas()`
    # tidak dijamin sama antar eksekusi (makro-F1 bergeser ~0,003).
    pdf = pdf.sort_values("series_id", kind="mergesort").reset_index(drop=True)

    logger.info("Data training dimuat: %s baris, %s kolom", f"{len(pdf):,}", len(pdf.columns))
    return pdf, stopwords


def build_representation(corpus: pd.Series) -> tuple[TfidfVectorizer, TruncatedSVD, np.ndarray, np.ndarray]:
    """
    Bangun ruang fitur bersama.

    TF-IDF 1-2 gram: bigram penting karena tag aslinya berupa frasa
    ("time loop", "weak strong"); unigram saja akan memecah maknanya.
    SVD/LSA kemudian memadatkannya ke 200 dimensi sehingga cosine similarity
    tahan terhadap sinonim dan jauh lebih murah dihitung saat inferensi.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,            # buang frasa yang hanya muncul di 1-2 novel (noise)
        max_df=0.85,         # buang frasa yang ada di hampir semua novel (tak informatif)
        max_features=30_000,
        sublinear_tf=True,   # redam dominasi tag yang berulang di satu dokumen
    )
    tfidf = vectorizer.fit_transform(corpus)
    logger.info("Matriks TF-IDF: %s x %s (sparsity %.4f%%)",
                f"{tfidf.shape[0]:,}", f"{tfidf.shape[1]:,}",
                100 * tfidf.nnz / (tfidf.shape[0] * tfidf.shape[1]))

    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
    embeddings = svd.fit_transform(tfidf)
    explained = float(svd.explained_variance_ratio_.sum())
    logger.info("SVD %d komponen | variansi terjelaskan: %.2f%%", SVD_COMPONENTS, explained * 100)

    # Normalisasi L2 -> dot product == cosine similarity (rekomendasi jadi murah).
    embeddings = normalize(embeddings).astype(np.float32)
    return vectorizer, svd, tfidf, embeddings


def build_genre_frame(pdf: pd.DataFrame, stopwords: list[str] | None = None) -> pd.DataFrame:
    """Tabel fitur Head A; dengan `stopwords` teks dibangun ulang dan sensornya dipulihkan."""
    from Model.pipeline.preprocessing import build_feature_frame

    return build_feature_frame(pdf, stopwords=stopwords)


def train_genre_tagger(
    pdf: pd.DataFrame, names: list | None = None, frame: pd.DataFrame | None = None
) -> tuple[dict, MultiLabelBinarizer, dict]:
    """
    Latih arsitektur genre tagger dan kembalikan seluruhnya.

    Pengguna memilih model mana yang dipakai di dashboard, jadi semuanya
    disimpan sebagai artefak — bukan hanya pemenangnya. Detail arsitektur,
    rekayasa fitur, augmentasi, dan penyetelan ambang ada di `Model/pipeline/`.
    """
    from Model.pipeline import train as pipeline_train
    from Model.pipeline.preprocessing import build_label_matrix

    y, kept, mlb = build_label_matrix(pdf["genre_list"], MIN_GENRE_SUPPORT)
    total_genres = len({g for row in pdf["genre_list"] if row is not None for g in row})
    logger.info("Genre dipakai: %d dari %d (ambang >= %d novel)", len(kept), total_genres, MIN_GENRE_SUPPORT)

    if frame is None:
        frame = build_genre_frame(pdf)
    dev_idx, test_idx = pipeline_train.split_indices(len(frame))
    logger.info(
        "Pembagian data: pengembangan=%s (ambang via OOF %d-fold), uji=%s",
        f"{len(dev_idx):,}", 3, f"{len(test_idx):,}",
    )

    artifacts, metrics = pipeline_train.train_all(frame, y, kept, dev_idx, test_idx, names=names)
    metrics.update(
        {
            "n_labels": len(kept),
            "n_train": int(len(dev_idx)),
            "n_test": int(len(test_idx)),
            "feature_columns": list(genre_models.FEATURE_COLUMNS),
        }
    )
    return artifacts, mlb, metrics


def train_clustering(embeddings: np.ndarray, vectorizer, tfidf, pdf: pd.DataFrame) -> tuple[KMeans, dict]:
    """
    Segmentasi katalog menjadi arketipe cerita.

    k dipilih lewat silhouette score, bukan ditebak. Silhouette dihitung pada
    subsampel 2.000 titik karena kompleksitasnya O(n^2): pada 39 ribu titik
    itu ~1,6 miliar jarak per nilai k, padahal yang diputuskan hanya nilai k.
    """
    scores = {}
    for k in KMEANS_K_GRID:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        assign = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, assign, sample_size=2000, random_state=RANDOM_STATE)
        scores[k] = round(float(score), 4)
        logger.info("  k=%2d | silhouette=%.4f | inertia=%.1f", k, score, km.inertia_)

    best_k = max(scores, key=scores.get)
    logger.info("HEAD C | k terpilih = %d (silhouette %.4f)", best_k, scores[best_k])

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    assignments = kmeans.fit_predict(embeddings)

    # Profil klaster: istilah dominan diambil dari rata-rata TF-IDF anggotanya,
    # jauh lebih akurat daripada mengurai centroid di ruang SVD (yang bisa negatif).
    vocab = np.array(vectorizer.get_feature_names_out())
    profiles = []
    for cluster_id in range(best_k):
        mask = assignments == cluster_id
        member_tfidf = tfidf[mask]
        mean_scores = np.asarray(member_tfidf.mean(axis=0)).ravel()
        top_terms = vocab[np.argsort(mean_scores)[::-1][:12]].tolist()

        subset = pdf[mask]
        genres = pd.Series([g for row in subset["genre_list"] for g in (row if row is not None else [])])
        profiles.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(mask.sum()),
                "top_terms": top_terms,
                "top_genres": genres.value_counts().head(5).to_dict(),
                "avg_weighted_rating": round(float(subset["weighted_rating"].mean(skipna=True)), 4),
                # 3 desimal: tampilan membulatkan lagi ke bab utuh, 1 desimal (654,5) membuatnya salah arah.
                "avg_chapters": round(float(subset["chapter_count"].mean(skipna=True)), 3),
                "example_titles": subset.nlargest(5, "rating_votes")["title"].tolist(),
            }
        )
        logger.info("  klaster %d (n=%d): %s", cluster_id, int(mask.sum()), ", ".join(top_terms[:6]))

    return kmeans, {"best_k": int(best_k), "silhouette_by_k": scores, "profiles": profiles}


def build_numeric_features(pdf: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Fitur non-teks untuk regressor.

    Fitur popularitas (`rating_votes`, `reading_list_count`) SENGAJA TIDAK
    dipakai: `weighted_rating` dihitung dari jumlah suara, sehingga
    memasukkannya berarti membocorkan target. Model ini harus menjawab
    "seberapa disukai novel dengan karakteristik seperti ini", bukan
    "berapa rating novel yang sudah ramai dinilai".
    """
    origin_dummies = pd.get_dummies(pdf["type_origin"].fillna("UNKNOWN"), prefix="origin")
    type_dummies = pd.get_dummies(pdf["type"].fillna("UNKNOWN"), prefix="type")

    frame = pd.DataFrame(
        {
            "log_chapters": np.log1p(pdf["chapter_count"].fillna(pdf["chapter_count"].median())),
            "n_tags": pdf["n_tags"].fillna(0),
            "token_count": pdf["token_count"].fillna(0),
            "year": pdf["year"].fillna(pdf["year"].median()),
            "is_completed": pdf["is_completed"].fillna(0),
            # Kecepatan rilis (r = -0,089 terhadap rating), di-log karena ekornya panjang. Bukan fitur popularitas:
            # ditentukan penulis/penerjemah, jadi tidak membocorkan target seperti `rating_votes`.
            "log_days_per_chapter": np.log1p(
                pdf["days_per_chapter"].fillna(pdf["days_per_chapter"].median())
                if "days_per_chapter" in pdf.columns
                else pd.Series(np.zeros(len(pdf)), index=pdf.index)
            ),
        }
    )
    frame = pd.concat([frame, origin_dummies, type_dummies], axis=1)
    return frame.to_numpy(dtype=np.float32), frame.columns.tolist()


def rating_texts(frame: pd.DataFrame) -> list[str]:
    """Teks masukan skor ridge Head D: korpus bersih + tag (sama seperti inferensi)."""
    return (frame["corpus_clean"].astype(str) + " " + frame["tags_clean"].astype(str)).tolist()


def train_rating_estimator(
    embeddings: np.ndarray, pdf: pd.DataFrame, frame: pd.DataFrame | None = None
) -> tuple[RatingModel | None, dict, list[str]]:
    """
    Regresi `rating_score` dari konten.

    Dilatih HANYA pada populasi ber-rating tepercaya (>= 10 suara), karena
    rating dari 1-2 suara adalah derau, bukan sinyal — melatih di atasnya
    mengajarkan model meniru kebisingan.

    HistGradientBoosting dipilih ketimbang regresi linear karena hubungan
    fitur-rating jelas non-linear (lihat Fase 3: rating naik lalu turun
    terhadap panjang novel) dan ia menangani ratusan fitur padat dengan cepat.
    Fitur & bobotnya dijelaskan di `Model/pipeline/rating.py`; `frame` adalah
    tabel teks Head A agar skor ridge melihat teks yang sama dengan inferensi.
    """
    if frame is None:
        frame = build_genre_frame(pdf)
    numeric, feature_names = build_numeric_features(pdf)
    mask = pdf["is_rated_reliable"].fillna(False).to_numpy(dtype=bool)
    rows = np.flatnonzero(mask)
    y = pdf.loc[mask, "rating_score"].to_numpy(dtype=np.float32)
    # Rating dari 500 suara lebih stabil daripada dari 10 suara.
    weights = np.log1p(pdf.loc[mask, "rating_votes"].fillna(0).to_numpy(dtype=np.float32))
    texts = np.asarray(rating_texts(frame), dtype=object)
    tags = pdf["tags"].fillna("").to_numpy(dtype=object)

    if len(y) < 200:
        logger.warning("HEAD D dilewati: populasi rating tepercaya terlalu kecil (%d).", len(y))
        return None, {"trained": False, "reason": "populasi terlalu kecil"}, feature_names

    # Pembagian sama dengan versi lama, jadi angka sebelum/sesudah memakai baris uji yang sama.
    pos_train, pos_test = train_test_split(
        np.arange(len(rows)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train, test = rows[pos_train], rows[pos_test]
    y_train, y_test = y[pos_train], y[pos_test]

    # Untuk evaluasi, TF-IDF + SVD di-fit tanpa baris uji, sama seperti judul baru di produksi
    # yang tidak ikut membentuk representasi bersama.
    held_out = np.setdiff1d(np.arange(len(frame)), test)
    eval_vectorizer, eval_svd, _, _ = build_representation(frame["corpus_clean"].iloc[held_out])
    eval_embeddings = normalize(
        eval_svd.transform(eval_vectorizer.transform(frame["corpus_clean"]))
    ).astype(np.float32)

    model = RatingModel(HGB_PARAMS, feature_names, random_state=RANDOM_STATE).fit(
        eval_embeddings[train], numeric[train], list(texts[train]), list(tags[train]), y_train,
        sample_weight=weights[pos_train],
    )
    pred = model.predict(eval_embeddings[test], numeric[test], list(texts[test]), list(tags[test]))

    # Pembanding: konfigurasi lama (embedding + numerik, tanpa bobot) pada baris latih yang sama.
    previous = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=6, l2_regularization=1.0,
        random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.15,
    ).fit(np.hstack([eval_embeddings, numeric])[train], y_train)
    pred_previous = previous.predict(np.hstack([eval_embeddings, numeric])[test])

    # Baseline = selalu menebak rata-rata. Model yang tidak mengalahkan ini
    # tidak layak dipakai, dan itu harus dilaporkan apa adanya.
    baseline = np.full_like(y_test, y_train.mean())
    metrics = {
        "trained": True,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
        "r2": round(float(r2_score(y_test, pred)), 4),
        "baseline_mae": round(float(mean_absolute_error(y_test, baseline)), 4),
        "baseline_rmse": round(float(np.sqrt(mean_squared_error(y_test, baseline))), 4),
        "previous_config": {
            "mae": round(float(mean_absolute_error(y_test, pred_previous)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred_previous))), 4),
            "r2": round(float(r2_score(y_test, pred_previous)), 4),
            "features": "embedding SVD + numerik, tanpa bobot sampel",
        },
        "features": {
            "embedding_svd": int(embeddings.shape[1]),
            "numeric": len(feature_names),
            "ridge_text_score": 1,
            "tag_multi_hot": model.n_tag_features,
        },
        "hgb_params": {k: v for k, v in HGB_PARAMS.items() if v is not None},
        "n_iter": int(model.hgb_.n_iter_),
        "sample_weight": "log1p(rating_votes)",
        "evaluation_representation": "TF-IDF + SVD di-fit tanpa baris uji",
        "target": "rating_score (populasi >= 10 suara)",
    }
    metrics["improvement_over_baseline_pct"] = round(
        100 * (metrics["baseline_mae"] - metrics["mae"]) / metrics["baseline_mae"], 2
    )
    logger.info(
        "HEAD D | MAE=%.4f (baseline %.4f, +%.1f%%; konfigurasi lama %.4f) | RMSE=%.4f | R2=%.4f",
        metrics["mae"], metrics["baseline_mae"], metrics["improvement_over_baseline_pct"],
        metrics["previous_config"]["mae"], metrics["rmse"], metrics["r2"],
    )

    model_full = RatingModel(HGB_PARAMS, feature_names, random_state=RANDOM_STATE).fit(
        embeddings[rows], numeric[rows], list(texts[rows]), list(tags[rows]), y, sample_weight=weights,
    )
    return model_full, metrics, feature_names


def evaluate_recommender(embeddings: np.ndarray, pdf: pd.DataFrame, k: int = 10) -> dict:
    """
    Ukur kualitas rekomendasi tanpa ground-truth perilaku.

    Proksi yang dipakai: *genre overlap@k* — berapa persen dari k tetangga
    terdekat yang berbagi minimal satu genre dengan novel acuan. Ini bukan
    pengganti uji A/B, tetapi cukup membuktikan ruang embedding menangkap
    kemiripan semantik dan bukan sekadar derau.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(pdf), size=min(500, len(pdf)), replace=False)
    genre_sets = [set(g) if g is not None else set() for g in pdf["genre_list"]]

    overlaps = []
    for idx in sample_idx:
        sims = embeddings @ embeddings[idx]
        sims[idx] = -np.inf
        top = np.argpartition(-sims, k)[:k]
        base = genre_sets[idx]
        if not base:
            continue
        overlaps.append(sum(1 for j in top if genre_sets[j] & base) / k)

    score = float(np.mean(overlaps)) if overlaps else 0.0
    logger.info("HEAD B | genre overlap@%d = %.4f (n sampel=%d)", k, score, len(overlaps))
    return {"genre_overlap_at_k": round(score, 4), "k": k, "n_sampled": len(overlaps)}


def _write_atomic(path: Path, write) -> None:
    """
    Tulis ke `<nama>.tmp` lalu ganti nama: pembaca (Streamlit) tidak pernah
    melihat berkas yang setengah tertulis, dan kegagalan di tengah tidak
    merusak versi sebelumnya.
    """
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def save_artifacts(**artifacts) -> None:
    """Simpan seluruh artefak ke direktori bind-mount agar dibaca Streamlit."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for name, obj in artifacts.items():
        path = ARTIFACT_DIR / f"{name}.joblib"
        if obj is None:
            # Berkas dari run lama (mis. regressor saat Head D kini dilewati)
            # dilatih di ruang fitur lama - jangan dibiarkan termuat.
            if path.is_file():
                path.unlink()
                logger.info("Artefak basi dihapus: %s", path.name)
            continue
        if name == "genre_models":
            from Model.pipeline.models import share_feature_steps

            shared = share_feature_steps(obj)
            if shared:
                logger.info("Langkah fitur identik dipakai bersama oleh: %s", ", ".join(shared))
        _write_atomic(path, lambda tmp, obj=obj: dump_artifact(obj, tmp))
        logger.info("Artefak tersimpan: %s (%.2f MB)", path.name, path.stat().st_size / 1_048_576)


def persist_run(
    pdf: pd.DataFrame,
    stopwords: list[str],
    vectorizer,
    svd,
    tfidf,
    embeddings: np.ndarray,
    genre_artifacts: dict,
    mlb: MultiLabelBinarizer,
    metrics_a: dict,
    metrics_b: dict,
    kmeans,
    metrics_c: dict,
    regressor,
    metrics_d: dict,
    numeric_features: list[str],
    upload: bool = True,
) -> dict:
    """
    Tulis seluruh artefak satu run Fase 4 dan kembalikan isi metrics.json.

    Dipakai `run()` supaya penyimpanan selalu atomik dan metrics.json ditulis paling akhir.
    """
    # Katalog untuk recommender (metadata ringan, tanpa kolom teks berat)
    catalog = pdf[
        ["series_id", "title", "url", "primary_genre", "genre_list", "tags", "type_origin",
         "year", "chapter_count", "rating_score", "rating_votes", "weighted_rating", "n_tags",
         "synopsis_snippet", "synopsis_word_count", "days_per_chapter"]
    ].copy()
    catalog["cluster_id"] = kmeans.labels_
    catalog["genre_list"] = catalog["genre_list"].apply(lambda g: list(g) if g is not None else [])

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    _write_atomic(ARTIFACT_DIR / "catalog.parquet", lambda tmp: catalog.to_parquet(tmp, index=False))
    # np.save menambahkan ".npy" bila nama tidak berakhiran itu - tulis lewat handle.
    def _save_npy(tmp):
        with open(tmp, "wb") as fh:
            np.save(fh, embeddings)
    _write_atomic(ARTIFACT_DIR / "embeddings.npy", _save_npy)
    logger.info("Katalog (%s baris) & embedding %s tersimpan.", f"{len(catalog):,}", embeddings.shape)

    save_artifacts(
        vectorizer=vectorizer,
        svd=svd,
        # Seluruh model genre disimpan sekaligus — pengguna dashboard memilih
        # mana yang dipakai, jadi yang kalah pun tetap harus tersedia.
        genre_models=genre_artifacts,
        genre_binarizer=mlb,
        kmeans=kmeans,
        rating_regressor=regressor,
    )

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_documents": int(len(pdf)),
        "representation": {
            "tfidf_features": int(tfidf.shape[1]),
            "svd_components": SVD_COMPONENTS,
            "explained_variance": round(float(svd.explained_variance_ratio_.sum()), 4),
        },
        "head_a_genre_tagger": metrics_a,
        "head_b_recommender": metrics_b,
        "head_c_clustering": metrics_c,
        "head_d_rating_estimator": metrics_d,
        "numeric_features": numeric_features,
        "stopwords": stopwords,
    }
    # metrics.json ditulis paling akhir: dashboard memakai waktu ubahnya sebagai
    # penanda bahwa satu set artefak sudah lengkap (lihat get_registry).
    _write_atomic(
        ARTIFACT_DIR / "metrics.json",
        lambda tmp: tmp.write_text(json.dumps(metrics, indent=2), encoding="utf-8"),
    )
    logger.info("Metrik & konfigurasi ditulis ke %s", ARTIFACT_DIR / "metrics.json")

    if upload:
        # Artefak juga disalin ke curated zone sebagai cadangan yang dapat diaudit.
        _upload_artifacts_to_lake()
    return metrics


def run() -> dict:
    log_section(logger, "fase 4 - model (arsitektur ai multi-head)")

    pdf, stopwords = load_training_frame()
    # Satu tabel teks (sensor dipulihkan) untuk keempat kepala, sama dengan jalur inferensi.
    frame = build_genre_frame(pdf, stopwords)

    logger.info(">> Membangun representasi bersama (TF-IDF -> SVD)")
    vectorizer, svd, tfidf, embeddings = build_representation(frame["corpus_clean"])

    # Kosong = DEFAULT_MODELS; PHASE4_MODELS=svm,rf membatasi model yang dilatih.
    names = ([n.strip() for n in os.environ.get("PHASE4_MODELS", "").split(",") if n.strip()]
             or list(genre_models.DEFAULT_MODELS))
    logger.info(">> HEAD A: genre tagger (%s, multi-label)", ", ".join(names))
    genre_artifacts, mlb, metrics_a = train_genre_tagger(pdf, names=names, frame=frame)

    logger.info(">> HEAD B: content-based recommender")
    metrics_b = evaluate_recommender(embeddings, pdf)

    logger.info(">> HEAD C: archetype clustering")
    kmeans, metrics_c = train_clustering(embeddings, vectorizer, tfidf, pdf)

    logger.info(">> HEAD D: rating estimator")
    regressor, metrics_d, numeric_features = train_rating_estimator(embeddings, pdf, frame)

    metrics = persist_run(
        pdf, stopwords, vectorizer, svd, tfidf, embeddings,
        genre_artifacts, mlb, metrics_a, metrics_b, kmeans, metrics_c,
        regressor, metrics_d, numeric_features,
    )

    logger.info("FASE 4 SELESAI")
    return metrics


def _upload_artifacts_to_lake() -> None:
    """Salin artefak model ke `curated-data/models/` (backup + auditability)."""
    from src.common.storage import upload_file

    settings = get_settings()
    for path in sorted(ARTIFACT_DIR.iterdir()):
        if path.is_file() and not path.name.startswith("_"):
            upload_file(path, settings.minio.bucket_curated, f"models/{path.name}", settings)


def main() -> int:
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        logger.error("FASE 4 GAGAL: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
