"""
Orkestrasi pelatihan & protokol evaluasi
Satu tempat yang memastikan setiap arsitektur diperlakukan sama persis —
tanpa itu, perbandingan antar model tidak bermakna.

Protokolnya:

  1. Data dibagi **80% pengembangan / 20% uji**. Set uji tidak disentuh sampai
     langkah terakhir.
  2. Untuk tiap kandidat hyperparameter, probabilitas *out-of-fold* dihitung
     lewat k-fold CV pada set pengembangan. Ambang per genre disetel di atas
     probabilitas OOF itu, lalu macro-F1 OOF menjadi skor pemilihan.
  3. Konfigurasi terbaik dilatih ulang pada SELURUH set pengembangan.
  4. Angka yang dilaporkan diukur pada set uji — sekali, di akhir.

Kenapa OOF dan bukan set validasi terpisah: menyisihkan 20% lagi berarti
~6.300 judul tidak ikut melatih, padahal genre langka seperti Shoujo Ai hanya
punya ~550 contoh positif di set pengembangan. OOF memberi estimasi yang sama
jujurnya tanpa mengorbankan data.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from Model.pipeline import evaluate as ev
from Model.pipeline import models as mdl
from Model.pipeline.config import (
    CV_FOLDS,
    FeatureConfig,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.common.logging_utils import get_logger

logger = get_logger("model.pipeline")


class FeatureCache:
    """
    Langkah `features` yang sudah di-fit, per potongan data dan per konfigurasi fitur.

    logreg, svm, dan rf memakai ColumnTransformer yang identik (stemming + tiga TF-IDF).
    Tanpa cache langkah itu di-fit 15 kali di OOF (5 kandidat x 3 fold) dan 3 kali di fit
    akhir, ~14 detik setiap kalinya, padahal hasilnya identik byte demi byte (artefak yang
    disimpan pun sudah berbagi satu objek, lihat `models.share_feature_steps`). Hanya
    berlaku di dalam satu `train_all`, karena kuncinya posisi baris, bukan isi frame.
    """

    def __init__(self) -> None:
        self._store: dict = {}

    def get(self, step, frame: pd.DataFrame, fit_rows: np.ndarray, key: tuple, held_rows=None):
        """(langkah ter-fit, X baris latih, X baris tahan) untuk salinan `step` yang di-fit di `fit_rows`."""
        cache_key = (key, joblib.hash(step))
        if cache_key not in self._store:
            fitted = clone(step)
            x_fit = fitted.fit_transform(frame.iloc[fit_rows])
            x_held = fitted.transform(frame.iloc[held_rows]) if held_rows is not None else None
            self._store[cache_key] = (fitted, x_fit, x_held)
        return self._store[cache_key]


def _split_features(estimator):
    """(langkah `features`, sisa pipeline) bila estimator Pipeline berawalan `features`; selain itu (None, None)."""
    if isinstance(estimator, Pipeline) and estimator.steps[0][0] == "features" and len(estimator.steps) > 1:
        return estimator.steps[0][1], estimator[1:]
    return None, None


def out_of_fold_probabilities(
    estimator, frame: pd.DataFrame, y: np.ndarray, folds: int = CV_FOLDS, cache: FeatureCache | None = None
) -> np.ndarray:
    """
    Probabilitas out-of-fold untuk seluruh set pengembangan.

    Setiap baris diprediksi oleh model yang TIDAK melihatnya saat dilatih,
    sehingga probabilitas ini bebas kebocoran dan layak dipakai untuk dua hal
    sekaligus: memilih hyperparameter dan menyetel ambang. Dengan `cache`,
    hasilnya sama dengan `cross_val_predict` tetapi fitur tiap fold di-fit sekali
    untuk semua kandidat.
    """
    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    features, head = _split_features(estimator)
    if cache is None or features is None:
        proba = cross_val_predict(estimator, frame, y, cv=splitter, method="predict_proba", n_jobs=1)
        if isinstance(proba, list):
            proba = np.column_stack([p[:, 1] for p in proba])
        return np.asarray(proba)

    proba = None
    for fold, (fit_rows, held_rows) in enumerate(splitter.split(frame)):
        _, x_fit, x_held = cache.get(features, frame, fit_rows, ("oof", folds, fold), held_rows)
        part = mdl.predict_proba_frame(clone(head).fit(x_fit, y[fit_rows]), x_held)
        if proba is None:
            proba = np.empty((len(frame), part.shape[1]), dtype=part.dtype)
        proba[held_rows] = part
    return proba


def split_indices(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Bagi indeks menjadi set pengembangan dan set uji (deterministik)."""
    return train_test_split(np.arange(n_rows), test_size=TEST_SIZE, random_state=RANDOM_STATE)


def train_single(
    name: str,
    frame: pd.DataFrame,
    y: np.ndarray,
    labels: list[str],
    dev_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_config: FeatureConfig | None = None,
    param_grid: list[dict] | None = None,
    cache: FeatureCache | None = None,
) -> tuple[object, np.ndarray, dict, np.ndarray, np.ndarray]:
    """Latih satu arsitektur: cari konfigurasi, setel ambang, uji sekali.

    Probabilitas OOF dan uji konfigurasi terpilih ikut dikembalikan untuk ensemble.
    """
    meta = mdl.MODEL_CATALOG[name]
    grid = param_grid if param_grid is not None else mdl.PARAM_GRID[name]
    x_dev, x_test = frame.iloc[dev_idx], frame.iloc[test_idx]
    y_dev, y_test = y[dev_idx], y[test_idx]

    logger.info("  [%s] %s — mencari konfigurasi terbaik ...", name, meta["label"])
    best = {"score": -1.0, "params": grid[0], "thresholds": None}

    for params in grid:
        candidate = mdl.build_model(name, feature_config, **params)
        prob_oof = out_of_fold_probabilities(candidate, x_dev, y_dev, cache=cache)
        thresholds = ev.tune_thresholds(y_dev, prob_oof, labels)
        score = ev.evaluate(y_dev, prob_oof, thresholds)["f1_macro"]
        logger.info("      %s -> makro-F1 OOF %.4f", params, score)
        if score > best["score"]:
            best = {"score": score, "params": params, "thresholds": thresholds, "prob_oof": prob_oof}

    logger.info("  [%s] konfigurasi terpilih: %s", name, best["params"])
    model = mdl.build_model(name, feature_config, **best["params"])
    features, head = _split_features(model)
    if cache is not None and features is not None:
        fitted, x_fit, _ = cache.get(features, x_dev, np.arange(len(x_dev)), ("dev",))
        model = Pipeline([("features", fitted), *clone(head).fit(x_fit, y_dev).steps])
    else:
        model.fit(x_dev, y_dev)
    # Dirampingkan sebelum diuji, jadi angka uji berasal dari model yang benar-benar disimpan.
    mdl.shrink_coefficients(model)

    thresholds = best["thresholds"]
    prob_test = mdl.predict_proba_frame(model, x_test)
    scores = ev.evaluate(y_test, prob_test, thresholds)
    # Pembanding: model yang sama dengan ambang seragam 0,5. Selisihnya
    # memperlihatkan kontribusi penyetelan ambang per genre.
    fixed = ev.evaluate(y_test, prob_test, np.full(len(labels), 0.5, dtype=np.float32))

    scores.update(
        {
            "f1_micro_fixed_threshold": fixed["f1_micro"],
            "f1_macro_fixed_threshold": fixed["f1_macro"],
            "f1_macro_oof": round(float(best["score"]), 4),
            "params": {k: str(v) for k, v in best["params"].items()},
            "threshold_mean": round(float(thresholds.mean()), 3),
            "f1_per_label": ev.per_label_scores(y_test, prob_test, thresholds, labels),
            **meta,
        }
    )
    logger.info(
        "  [%s] UJI: F1 makro=%.4f mikro=%.4f P@3=%.4f | ambang seragam 0,5: makro=%.4f",
        name, scores["f1_macro"], scores["f1_micro"], scores["precision_at_3"], fixed["f1_macro"],
    )
    return model, thresholds, scores, best["prob_oof"], prob_test


def blend_members(
    probs: dict, y: np.ndarray, labels: list[str], dev_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[dict, np.ndarray, dict]:
    """
    Gabungan berbobot anggota ensemble di ruang log-odds (`ev.blend_probabilities`).

    Bobot dipilih di grid 0,0-1,0 atas probabilitas OOF, persis seperti hyperparameter.
    """
    first, second = mdl.ENSEMBLE_MEMBERS
    space = mdl.ENSEMBLE_SPACE
    y_dev, y_test = y[dev_idx], y[test_idx]
    best = None
    for weight in np.round(np.arange(0.0, 1.0001, 0.1), 1):
        mixed = ev.blend_probabilities([(weight, probs[first][0]), (1 - weight, probs[second][0])], space)
        thresholds = ev.tune_thresholds(y_dev, mixed, labels)
        score = ev.evaluate(y_dev, mixed, thresholds)["f1_macro"]
        if best is None or score > best[0]:
            best = (score, float(weight), thresholds)
    score, weight, thresholds = best
    weights = {first: weight, second: round(1.0 - weight, 1)}

    prob_test = ev.blend_probabilities([(weight, probs[first][1]), (1 - weight, probs[second][1])], space)
    scores = ev.evaluate(y_test, prob_test, thresholds)
    fixed = ev.evaluate(y_test, prob_test, np.full(len(labels), 0.5, dtype=np.float32))
    scores.update(
        {
            "f1_micro_fixed_threshold": fixed["f1_micro"],
            "f1_macro_fixed_threshold": fixed["f1_macro"],
            "f1_macro_oof": round(float(score), 4),
            "params": {"weights": str(weights), "space": space},
            "threshold_mean": round(float(thresholds.mean()), 3),
            "f1_per_label": ev.per_label_scores(y_test, prob_test, thresholds, labels),
            **mdl.MODEL_CATALOG["ensemble"],
        }
    )
    logger.info("  [ensemble] bobot %s | OOF %.4f | UJI: F1 makro=%.4f mikro=%.4f",
                weights, score, scores["f1_macro"], scores["f1_micro"])
    return weights, thresholds, scores


def train_all(
    frame: pd.DataFrame,
    y: np.ndarray,
    labels: list[str],
    dev_idx: np.ndarray,
    test_idx: np.ndarray,
    names: list[str] | None = None,
    feature_config: FeatureConfig | None = None,
) -> tuple[dict, dict]:
    """Latih seluruh arsitektur pada pembagian data yang identik."""
    artifacts, metrics, probs = {}, {}, {}
    cache = FeatureCache()
    requested = list(names or mdl.MODEL_CATALOG)
    for name in requested:
        if name == "ensemble":
            continue
        model, thresholds, scores, prob_oof, prob_test = train_single(
            name, frame, y, labels, dev_idx, test_idx, feature_config, cache=cache
        )
        # `labels` ikut disimpan per model: penggabungan latih-parsial
        # (Model/train.py --models) harus bisa memastikan semua model berbagi
        # ruang label yang sama dengan genre_binarizer.
        artifacts[name] = {"pipeline": model, "thresholds": thresholds, "labels": list(labels)}
        metrics[name] = scores
        probs[name] = (prob_oof, prob_test)

    if "ensemble" in requested:
        missing = [m for m in mdl.ENSEMBLE_MEMBERS if m not in probs]
        if missing:
            logger.warning("Ensemble dilewati: %s tidak dilatih pada run ini.", ", ".join(missing))
        else:
            # Hanya bobot yang disimpan; pipeline anggota sudah ada di artefak masing-masing.
            weights, thresholds, scores = blend_members(probs, y, labels, dev_idx, test_idx)
            artifacts["ensemble"] = {"members": weights, "space": mdl.ENSEMBLE_SPACE,
                                     "thresholds": thresholds, "labels": list(labels)}
            metrics["ensemble"] = scores

    # Dipilih dari skor OOF set dev, bukan skor uji: memilih model berdasarkan
    # set uji membuat skor uji pemenang bias optimistis (set uji ikut seleksi).
    best = max(metrics, key=lambda k: metrics[k]["f1_macro_oof"])
    logger.info("Model terbaik menurut makro-F1 OOF: %s (OOF %.4f, uji %.4f)",
                best, metrics[best]["f1_macro_oof"], metrics[best]["f1_macro"])
    return artifacts, {"models": metrics, "best_model": best, "labels": labels}
