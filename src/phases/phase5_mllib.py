"""
Fase 5 - Head A dan Head D versi Spark MLlib, dibandingkan dengan scikit-learn
Fase 4 melatih model di scikit-learn: data disaring Spark, lalu ~40 ribu baris
di-`toPandas()` ke driver. Modul ini menjawab pertanyaan yang tersisa: apa yang
berubah bila pelatihannya sendiri juga dikerjakan Spark MLlib, terdistribusi
di worker?

Supaya perbandingannya adil, kedua kerangka memakai protokol yang sama persis:

  * aturan pembagian sama dengan Fase 4 (urut `series_id`, `train_test_split`
    seed 42, 20% uji) dan kedua kerangka dilatih ulang pada snapshot zona clean
    yang sama, sehingga baris uji MLlib = baris uji scikit-learn;
  * 35 genre yang sama (dibaca dari `Model/artifacts/metrics.json`);
  * 15% set pengembangan disisihkan sebagai set validasi untuk menyetel ambang
    per genre; model dilatih pada 85% sisanya;
  * metrik dihitung oleh fungsi yang sama (`Model.pipeline.evaluate`).

Protokol ini berbeda dari Fase 4 (ambang dari probabilitas out-of-fold 3-fold),
karena k-fold untuk 35 x 2 model MLlib terlalu mahal pada worker 2 core. Karena
itu scikit-learn dilatih ulang di sini dengan protokol validasi yang sama;
angka Fase 4 tetap dilaporkan terpisah sebagai acuan.

Jalankan:
    docker compose exec pipeline python -m src.phases.phase5_mllib
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC, LogisticRegression
from pyspark.ml.feature import (
    IDF,
    CountVectorizer,
    NGram,
    Normalizer,
    OneHotEncoder,
    RegexTokenizer,
    SQLTransformer,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression as SkLogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from Model.pipeline import evaluate as ev
from Model.pipeline import models as mdl
from Model.pipeline.config import FeatureConfig, RANDOM_STATE, TEST_SIZE
from Model.pipeline.preprocessing import build_feature_frame
from src.common.config import get_settings
from src.common.logging_utils import get_logger, log_section
from src.common.spark_session import spark_session

logger = get_logger("phase5.mllib")

PROJECT_DIR = Path("/opt/project")
METRICS_PATH = PROJECT_DIR / "Model/artifacts/metrics.json"
REPORT_PATH = PROJECT_DIR / "Doc/generated/mllib_comparison.json"
VALIDATION_SHARE = 0.15

# Padanan regularisasi dengan Fase 4: regParam = 1 / (C * n_latih), karena scikit-learn meminimalkan
# C * sum(loss) + ||w||^2 / 2 dan Spark mean(loss) + regParam * ||w||^2 / 2.
SK_PARAMS = {"logreg": {"C": 4.0, "class_weight": None}, "svm": {"C": 0.3, "class_weight": "balanced"}}

#: Fitur scikit-learn yang dibandingkan dengan MLlib (40k/40k/10k, bobot blok lama), dikunci
#: di sini karena bawaan FeatureConfig berubah pada revisi Fase 4 2026-10-15.
SK_FEATURES = FeatureConfig(
    corpus_max_features=40_000, synopsis_max_features=40_000, tags_max_features=10_000,
    weights={"corpus": 1.0, "synopsis": 1.0, "tags": 1.6, "meta": 0.6, "embedding": 1.0},
)

TOP_TAGS_FOR_RATING = 150


def _assign_splits(pdf: pd.DataFrame) -> pd.DataFrame:
    """Tandai baris dev/val/test dengan aturan yang sama seperti Fase 4."""
    pdf = pdf.sort_values("series_id", kind="mergesort").reset_index(drop=True)
    dev_idx, test_idx = train_test_split(np.arange(len(pdf)), test_size=TEST_SIZE, random_state=RANDOM_STATE)
    fit_idx, val_idx = train_test_split(dev_idx, test_size=VALIDATION_SHARE, random_state=RANDOM_STATE)
    split = np.empty(len(pdf), dtype=object)
    split[fit_idx], split[val_idx], split[test_idx] = "fit", "val", "test"
    pdf["split"] = split
    return pdf


def load_frame(spark) -> tuple[DataFrame, pd.DataFrame]:
    """Baca zona clean, tandai split, kembalikan versi Spark dan pandas-nya."""
    settings = get_settings()
    clean = spark.read.parquet(settings.minio.s3a_uri(settings.minio.bucket_clean, "novels_clean"))
    selected = clean.filter(F.col("has_text_signal")).select(
        "series_id", "title", "corpus_clean", "synopsis_clean", "synopsis", "tags", "type_origin", "type",
        "genre_list", "rating_score", "is_rated_reliable", "chapter_count", "n_tags", "token_count",
        "year", "is_completed", "days_per_chapter",
    )
    pdf = _assign_splits(selected.toPandas())
    split_sdf = spark.createDataFrame(pdf[["series_id", "split"]])
    sdf = (
        selected.join(split_sdf, on="series_id")
        .withColumn("tags_text", F.lower(F.regexp_replace(F.coalesce(F.col("tags"), F.lit("")), ",", " ")))
        .withColumn("corpus_text", F.coalesce(F.col("corpus_clean"), F.lit("")))
        .withColumn("synopsis_text", F.coalesce(F.col("synopsis_clean"), F.lit("")))
        .withColumn("type_origin", F.coalesce(F.col("type_origin"), F.lit("UNKNOWN")))
        .withColumn("type", F.coalesce(F.col("type"), F.lit("UNKNOWN")))
    )
    counts = pdf["split"].value_counts().to_dict()
    logger.info("Split: fit=%s val=%s test=%s", counts.get("fit"), counts.get("val"), counts.get("test"))
    return sdf, pdf


def _text_block(column: str, prefix: str, vocab: int, min_df: int) -> list:
    """
    Satu blok TF-IDF 1-2 gram, padanan `text_branch` di Model/pipeline/features.py.

    `Normalizer` wajib: TfidfVectorizer menormalisasi L2 tiap dokumen, sedangkan
    IDF Spark tidak. Tanpa normalisasi, norma vektor jauh lebih besar sehingga
    regParam yang setara di atas kertas menjadi regularisasi yang terlalu lemah
    dan Logistic Regression overfit (log loss latih 0,0007, uji 0,97 pada uji coba).
    """
    return [
        RegexTokenizer(inputCol=column, outputCol=f"{prefix}_words", pattern=r"\s+", minTokenLength=2),
        NGram(n=2, inputCol=f"{prefix}_words", outputCol=f"{prefix}_bigrams"),
        SQLTransformer(statement=f"SELECT *, concat({prefix}_words, {prefix}_bigrams) AS {prefix}_terms FROM __THIS__"),
        CountVectorizer(inputCol=f"{prefix}_terms", outputCol=f"{prefix}_tf", vocabSize=vocab, minDF=min_df, maxDF=0.85),
        IDF(inputCol=f"{prefix}_tf", outputCol=f"{prefix}_idf"),
        Normalizer(inputCol=f"{prefix}_idf", outputCol=f"{prefix}_vec", p=2.0),
    ]


def text_feature_pipeline() -> Pipeline:
    """
    Padanan blok fitur Fase 4 dengan komponen bawaan MLlib: corpus, sinopsis,
    tag, dan one-hot metadata.

    Tiga hal tidak punya padanan bawaan dan sengaja tidak ditambal dengan UDF:
    stemming Snowball, `sublinear_tf`, dan bobot per blok (tag 1,6, metadata
    0,6). Selisih skor yang muncul karenanya ikut menjadi temuan perbandingan.
    """
    stages = [
        *_text_block("corpus_text", "corpus", 40_000, 3),
        *_text_block("synopsis_text", "synopsis", 40_000, 3),
        *_text_block("tags_text", "tag", 10_000, 2),
        StringIndexer(inputCols=["type_origin", "type"], outputCols=["origin_idx", "type_idx"], handleInvalid="keep"),
        OneHotEncoder(inputCols=["origin_idx", "type_idx"], outputCols=["origin_vec", "type_vec"], handleInvalid="keep"),
        VectorAssembler(
            inputCols=["corpus_vec", "synopsis_vec", "tag_vec", "origin_vec", "type_vec"], outputCol="features"
        ),
    ]
    return Pipeline(stages=stages)


def _collect_scores(predictions: DataFrame, column: str, order: pd.Series) -> np.ndarray:
    """Ambil satu skor per baris, disusun mengikuti urutan `order` (series_id)."""
    frame = predictions.select("series_id", column).toPandas().set_index("series_id")
    return frame.loc[order.values, column].to_numpy(dtype=np.float64)


def _platt(val_margin: np.ndarray, val_true: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """Kalibrasi sigmoid satu dimensi untuk margin LinearSVC (padanan CalibratedClassifierCV)."""
    calib = SkLogisticRegression(C=1e6).fit(val_margin.reshape(-1, 1), val_true)
    return [calib.predict_proba(m.reshape(-1, 1))[:, 1] for m in (val_margin, *others)]


def mean_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Binary cross-entropy rata-rata atas 35 genre; loss yang sama untuk kedua kerangka."""
    prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    per_label = -(y_true * np.log(prob) + (1 - y_true) * np.log(1 - prob)).mean(axis=0)
    return round(float(per_label.mean()), 4)


def train_mllib_genre(sdf: DataFrame, pdf: pd.DataFrame, labels: list[str]) -> dict:
    """Latih 35 klasifikator biner per genre untuk LR dan LinearSVC, ukur di set uji."""
    started = time.perf_counter()
    feature_model = text_feature_pipeline().fit(sdf.filter(F.col("split") == "fit"))
    featurized = feature_model.transform(sdf).select("series_id", "split", "genre_list", "features").cache()
    n_features = featurized.first()["features"].size
    fit_df = featurized.filter(F.col("split") == "fit").cache()
    n_fit = fit_df.count()
    feature_seconds = time.perf_counter() - started
    logger.info("Fitur MLlib: %d dimensi, %d baris latih (%.1f dtk)", n_features, n_fit, feature_seconds)

    fit_ids = pdf.loc[pdf["split"] == "fit", "series_id"]
    val_ids = pdf.loc[pdf["split"] == "val", "series_id"]
    test_ids = pdf.loc[pdf["split"] == "test", "series_id"]
    y_val = _label_matrix(pdf.loc[pdf["split"] == "val", "genre_list"], labels)
    y_test = _label_matrix(pdf.loc[pdf["split"] == "test", "genre_list"], labels)
    y_fit = _label_matrix(pdf.loc[pdf["split"] == "fit", "genre_list"], labels)

    results = {}
    for kind in ("logreg", "svm"):
        reg_param = 1.0 / (SK_PARAMS[kind]["C"] * n_fit)
        prob_val = np.zeros((len(val_ids), len(labels)))
        prob_test = np.zeros((len(test_ids), len(labels)))
        prob_fit = np.zeros((len(fit_ids), len(labels)))
        histories = {}
        fit_seconds = 0.0
        for j, genre in enumerate(labels):
            labelled = fit_df.withColumn("label", F.array_contains("genre_list", genre).cast("double"))
            tick = time.perf_counter()
            if kind == "logreg":
                model = LogisticRegression(maxIter=100, regParam=reg_param, standardization=False).fit(labelled)
                fit_seconds += time.perf_counter() - tick
                scored = model.transform(featurized).withColumn("score", vector_to_array("probability")[1])
            else:
                # class_weight="balanced" versi scikit-learn: bobot n / (2 * n_kelas).
                positives = labelled.filter(F.col("label") == 1.0).count()
                w_pos, w_neg = n_fit / (2.0 * max(positives, 1)), n_fit / (2.0 * max(n_fit - positives, 1))
                weighted = labelled.withColumn("weight", F.when(F.col("label") == 1.0, w_pos).otherwise(w_neg))
                model = LinearSVC(maxIter=100, regParam=reg_param, standardization=False, weightCol="weight").fit(weighted)
                fit_seconds += time.perf_counter() - tick
                scored = model.transform(featurized).withColumn("score", vector_to_array("rawPrediction")[1])
            # Nilai fungsi objektif per iterasi L-BFGS/OWLQN: loss latih + suku regularisasi.
            # LinearSVCModel.summary adalah method, LogisticRegressionModel.summary adalah properti.
            summary = model.summary() if callable(model.summary) else model.summary
            histories[genre] = [round(float(v), 6) for v in summary.objectiveHistory]
            scored = scored.select("series_id", "score").cache()
            fit_score, val_score, test_score = (_collect_scores(scored, "score", ids) for ids in (fit_ids, val_ids, test_ids))
            scored.unpersist()
            if kind == "svm":
                val_score, fit_score, test_score = _platt(val_score, y_val[:, j], fit_score, test_score)
            prob_fit[:, j], prob_val[:, j], prob_test[:, j] = fit_score, val_score, test_score
            logger.info("  [mllib-%s] %2d/%d %s", kind, j + 1, len(labels), genre)

        results[kind] = _score_block(y_val, prob_val, y_test, prob_test, labels)
        results[kind].update({
            "fit_seconds": round(fit_seconds, 1),
            "reg_param": reg_param,
            "log_loss_train": mean_log_loss(y_fit, prob_fit),
            "iterations": {g: len(h) - 1 for g, h in histories.items()},
            "objective_history": histories,
        })
        logger.info("MLlib %s: makro-F1 uji %.4f, mikro-F1 %.4f, latih %.0f dtk",
                    kind, results[kind]["f1_macro"], results[kind]["f1_micro"], fit_seconds)

    featurized.unpersist()
    return {"feature_seconds": round(feature_seconds, 1), "n_features": int(n_features), "models": results}


def _label_matrix(genre_lists: pd.Series, labels: list[str]) -> np.ndarray:
    rows = [set(g) if g is not None else set() for g in genre_lists]
    return np.array([[1 if lab in row else 0 for lab in labels] for row in rows], dtype=int)


def _score_block(y_val, prob_val, y_test, prob_test, labels) -> dict:
    thresholds = ev.tune_thresholds(y_val, prob_val, labels)
    scores = ev.evaluate(y_test, prob_test, thresholds)
    fixed = ev.evaluate(y_test, prob_test, np.full(len(labels), 0.5))
    scores.update(
        {
            "f1_macro_fixed_threshold": fixed["f1_macro"],
            "f1_micro_fixed_threshold": fixed["f1_micro"],
            "f1_per_label": ev.per_label_scores(y_test, prob_test, thresholds, labels),
            "log_loss_val": mean_log_loss(y_val, prob_val),
            "log_loss_test": mean_log_loss(y_test, prob_test),
        }
    )
    return scores


def train_sklearn_genre(pdf: pd.DataFrame, labels: list[str]) -> dict:
    frame = build_feature_frame(pdf)
    y = _label_matrix(pdf["genre_list"], labels)
    masks = {name: (pdf["split"] == name).to_numpy() for name in ("fit", "val", "test")}

    results = {}
    for kind, params in SK_PARAMS.items():
        model = mdl.build_model(kind, SK_FEATURES, **params)
        tick = time.perf_counter()
        model.fit(frame[masks["fit"]], y[masks["fit"]])
        fit_seconds = time.perf_counter() - tick
        prob_val = mdl.predict_proba_frame(model, frame[masks["val"]])
        prob_test = mdl.predict_proba_frame(model, frame[masks["test"]])
        prob_fit = mdl.predict_proba_frame(model, frame[masks["fit"]])
        results[kind] = _score_block(y[masks["val"]], prob_val, y[masks["test"]], prob_test, labels)
        results[kind]["fit_seconds"] = round(fit_seconds, 1)
        results[kind]["log_loss_train"] = mean_log_loss(y[masks["fit"]], prob_fit)
        logger.info("scikit-learn %s: makro-F1 uji %.4f, mikro-F1 %.4f, latih %.0f dtk",
                    kind, results[kind]["f1_macro"], results[kind]["f1_micro"], fit_seconds)
    return results


def rating_features(sdf: DataFrame, pdf: pd.DataFrame, labels: list[str]) -> tuple[DataFrame, list[str]]:
    """
    Fitur tabular Head D: numerik Fase 4 + multi-hot genre + 150 tag tersering.

    Fase 4 memakai embedding SVD 200 dimensi dari scikit-learn. Untuk
    perbandingan kerangka, kedua sisi harus melihat fitur yang sama, jadi
    embedding diganti fitur yang bisa dibangun murni dengan operasi Spark.
    """
    reliable = pdf[pdf["is_rated_reliable"].fillna(False)]
    tag_counts = (
        reliable["tags"].fillna("").str.lower().str.split(",").explode().str.strip().replace("", np.nan).dropna()
        .value_counts()
    )
    top_tags = tag_counts.head(TOP_TAGS_FOR_RATING).index.tolist()

    med_chapters = float(reliable["chapter_count"].median())
    med_year = float(reliable["year"].median())
    med_days = float(reliable["days_per_chapter"].median())
    tag_array = F.transform(F.split(F.lower(F.coalesce(F.col("tags"), F.lit(""))), ","), lambda t: F.trim(t))

    cols = {
        "log_chapters": F.log1p(F.coalesce(F.col("chapter_count").cast("double"), F.lit(med_chapters))),
        "n_tags": F.coalesce(F.col("n_tags").cast("double"), F.lit(0.0)),
        "token_count": F.coalesce(F.col("token_count").cast("double"), F.lit(0.0)),
        "year": F.coalesce(F.col("year").cast("double"), F.lit(med_year)),
        "is_completed": F.coalesce(F.col("is_completed").cast("double"), F.lit(0.0)),
        "log_days_per_chapter": F.log1p(F.coalesce(F.col("days_per_chapter").cast("double"), F.lit(med_days))),
    }
    for origin in sorted(pdf["type_origin"].fillna("UNKNOWN").unique()):
        cols[f"origin_{origin}"] = (F.col("type_origin") == origin).cast("double")
    for kind in sorted(pdf["type"].fillna("UNKNOWN").unique()):
        cols[f"type_{kind}"] = (F.col("type") == kind).cast("double")
    for genre in labels:
        cols[f"genre_{genre}"] = F.array_contains("genre_list", genre).cast("double")
    for i, tag in enumerate(top_tags):
        cols[f"tag_{i:03d}"] = F.array_contains(tag_array, tag).cast("double")

    names = list(cols)
    out = sdf.filter(F.coalesce(F.col("is_rated_reliable"), F.lit(False))).select(
        "series_id", F.col("rating_score").cast("double").alias("label"), *[expr.alias(n) for n, expr in cols.items()]
    )
    return out, names


def train_rating(sdf: DataFrame, pdf: pd.DataFrame, labels: list[str]) -> dict:
    table, names = rating_features(sdf, pdf, labels)
    data = table.toPandas().sort_values("series_id", kind="mergesort").reset_index(drop=True)
    train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=TEST_SIZE, random_state=RANDOM_STATE)
    data["split"] = "train"
    data.loc[test_idx, "split"] = "test"
    train, test = data.iloc[train_idx], data.iloc[test_idx]
    baseline = np.full(len(test), train["label"].mean())

    def report(pred, seconds):
        return {
            "mae": round(float(mean_absolute_error(test["label"], pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(test["label"], pred))), 4),
            "r2": round(float(r2_score(test["label"], pred)), 4),
            "fit_seconds": round(seconds, 1),
        }

    tick = time.perf_counter()
    sk_model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=6, l2_regularization=1.0,
        random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.15,
    ).fit(train[names].to_numpy(), train["label"].to_numpy())
    sk = report(sk_model.predict(test[names].to_numpy()), time.perf_counter() - tick)
    # Kurva loss (MSE) per iterasi boosting, dihitung ulang dengan staged_predict supaya
    # satuannya sama dengan evaluateEachIteration milik Spark.
    sk["mse_train_curve"] = [round(float(mean_squared_error(train["label"], p)), 5)
                             for p in sk_model.staged_predict(train[names].to_numpy())]
    sk["mse_test_curve"] = [round(float(mean_squared_error(test["label"], p)), 5)
                            for p in sk_model.staged_predict(test[names].to_numpy())]
    sk["n_iter"] = int(sk_model.n_iter_)

    spark = sdf.sparkSession
    # GBT memperpanjang lineage RDD setiap pohon; tanpa direktori checkpoint, ratusan pohon
    # berakhir StackOverflowError di executor. Dengan direktori ini checkpointInterval=10
    # bawaan GBTRegressor aktif.
    settings = get_settings()
    spark.sparkContext.setCheckpointDir(settings.minio.s3a_uri(settings.minio.bucket_curated, "checkpoints/phase5"))
    split_sdf = spark.createDataFrame(data[["series_id", "split"]])
    assembled = VectorAssembler(inputCols=names, outputCol="features").transform(table.join(split_sdf, "series_id")).cache()
    train_sdf = assembled.filter(F.col("split") == "train").withColumn("is_val", F.rand(RANDOM_STATE) < 0.15).cache()
    train_sdf.count()
    test_sdf = assembled.filter(F.col("split") == "test")

    # Early stopping MLlib tanpa "patience": validationTol 0 berhenti pada kenaikan galat validasi pertama, dan bawaan 0,01 berhenti di pohon pertama.
    # Model pembanding memakai jumlah pohon yang dipilih early stopping HistGradientBoosting.
    early = GBTRegressor(maxIter=400, maxDepth=6, stepSize=0.06, seed=RANDOM_STATE,
                         validationIndicatorCol="is_val", validationTol=0.0).fit(train_sdf)
    early_trees = early.getNumTrees

    tick = time.perf_counter()
    gbt = GBTRegressor(maxIter=sk["n_iter"], maxDepth=6, stepSize=0.06, seed=RANDOM_STATE).fit(train_sdf)
    gbt_seconds = time.perf_counter() - tick
    pred = gbt.transform(test_sdf).select("series_id", "prediction").toPandas()
    pred = pred.set_index("series_id").loc[test["series_id"].values, "prediction"].to_numpy()
    ml = report(pred, gbt_seconds)
    ml["mse_train_curve"] = [round(float(v), 5) for v in gbt.evaluateEachIteration(train_sdf, "squared")]
    ml["mse_test_curve"] = [round(float(v), 5) for v in gbt.evaluateEachIteration(test_sdf, "squared")]
    ml["n_iter"] = gbt.getNumTrees
    ml["early_stopping_trees"] = early_trees
    train_sdf.unpersist()
    assembled.unpersist()

    result = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": len(names),
        "baseline_mae": round(float(mean_absolute_error(test["label"], baseline)), 4),
        "baseline_rmse": round(float(np.sqrt(mean_squared_error(test["label"], baseline))), 4),
        "sklearn_hgb": sk,
        "mllib_gbt": ml,
    }
    logger.info("Head D: HGB MAE %.4f (%.1f dtk) | GBT MLlib MAE %.4f (%.1f dtk) | baseline %.4f",
                sk["mae"], sk["fit_seconds"], ml["mae"], ml["fit_seconds"], result["baseline_mae"])
    return result


def run() -> dict:
    log_section(logger, "fase 5 - spark mllib vs scikit-learn")
    phase4 = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    labels = phase4["head_a_genre_tagger"]["labels"]

    with spark_session("phase5-mllib") as spark:
        sdf, pdf = load_frame(spark)
        sdf = sdf.cache()
        conf = spark.sparkContext.getConf()
        cluster = {
            "master": conf.get("spark.master"),
            "default_parallelism": spark.sparkContext.defaultParallelism,
            "executor_memory": conf.get("spark.executor.memory", ""),
        }
        mllib_genre = train_mllib_genre(sdf, pdf, labels)
        rating = train_rating(sdf, pdf, labels)

    sklearn_genre = train_sklearn_genre(pdf, labels)

    split_counts = pdf["split"].value_counts().to_dict()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_documents": int(len(pdf)),
        "split": {k: int(v) for k, v in split_counts.items()},
        "labels": labels,
        "spark": cluster,
        "head_a": {"mllib": mllib_genre, "sklearn": sklearn_genre},
        "head_d": rating,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Laporan perbandingan ditulis ke %s", REPORT_PATH)
    return report


def main() -> int:
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        logger.error("FASE 5 GAGAL: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
