"""
Estimasi rating (Head D)
Gradient boosting di atas empat kelompok fitur:

1. Embedding bersama TF-IDF -> SVD 200 dimensi (dipakai juga Head B/C).
2. Fitur numerik non-popularitas (bab, tahun, kecepatan rilis, asal, format).
3. Skor ridge teks: regresi ridge di TF-IDF penuh, out-of-fold untuk baris latih
   (stacked generalization, Wolpert 1992). Mengembalikan frasa jarang yang
   dibuang SVD.
4. Tag multi-hot: satu kolom per tag yang dipakai minimal 50 judul.

Bobot sampel log(1 + jumlah suara): rata-rata dari 500 suara lebih stabil
daripada dari 10 suara (weighted least squares).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from Model.pipeline.config import N_JOBS

#: Sapuan alpha 1 / 3 / 10: CV-MAE setelah ditumpuk 0,3349 / 0,3323 / 0,3322; 3 dipilih
#: karena ridge tunggalnya paling akurat (0,3378 vs 0,3440).
RIDGE_ALPHA = 3.0
RIDGE_FOLDS = 5
MIN_TAG_COUNT = 50

#: Hasil TPE (Bergstra dkk. 2011, via Optuna) 25 trial, MAE 3-fold CV di set latih saja.
HGB_PARAMS = {
    "max_iter": 800,
    "learning_rate": 0.0212,
    "max_leaf_nodes": 33,
    "max_depth": None,
    "min_samples_leaf": 43,
    "l2_regularization": 0.00144,
    "max_features": 0.204,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 20,
}


def ridge_text_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.85, max_features=200_000,
                           sublinear_tf=True, dtype=np.float32)


def tag_tokens(tags) -> list[str]:
    """Setiap tag -> satu token utuh ('Weak to Strong' -> 'tag_weak_to_strong')."""
    import re

    from src.common.text_prep import uncensor

    slug = re.compile(r"[^a-z0-9]+")
    out = []
    for row in pd.Series(tags).fillna("").astype(str):
        parts = (slug.sub("_", uncensor(t.strip().lower())).strip("_") for t in row.split(","))
        out.append(" ".join("tag_" + p for p in parts if p))
    return out


def _ridge_fold(texts: pd.Series, y: np.ndarray, train: np.ndarray, held: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vec = ridge_text_vectorizer()
    model = Ridge(alpha=RIDGE_ALPHA).fit(vec.fit_transform(texts.iloc[train]), y[train])
    return held, model.predict(vec.transform(texts.iloc[held]))


def ridge_oof(texts: list[str], y: np.ndarray, folds: int = RIDGE_FOLDS, seed: int = 42,
              n_jobs: int = N_JOBS) -> np.ndarray:
    """Skor ridge out-of-fold untuk baris latih (fitur level-2 yang jujur).

    Fold dikerjakan di proses terpisah: tokenisasi TF-IDF berjalan di Python murni
    (terkunci GIL), jadi thread tidak membantu. Setiap fold deterministik, sehingga
    hasilnya sama dengan versi berurutan; pada 18.604 judul ~32 dtk menjadi ~10 dtk.
    """
    texts = pd.Series(texts).reset_index(drop=True)
    out = np.zeros(len(texts), dtype=np.float32)
    splits = KFold(folds, shuffle=True, random_state=seed).split(texts)
    parts = Parallel(n_jobs=max(1, min(folds, n_jobs)))(
        delayed(_ridge_fold)(texts, y, train, held) for train, held in splits
    )
    for held, pred in parts:
        out[held] = pred
    return out


class RatingModel:
    """Head D lengkap: menyusun matriks fitur sendiri, lalu memprediksi."""

    def __init__(self, hgb_params: dict, numeric_features: list[str], random_state: int = 42):
        self.hgb_params = dict(hgb_params)
        self.numeric_features = list(numeric_features)
        self.random_state = random_state

    def fit(self, embeddings, numeric, texts, tags, y, sample_weight=None):
        self.text_vectorizer_ = ridge_text_vectorizer()
        self.ridge_ = Ridge(alpha=RIDGE_ALPHA).fit(self.text_vectorizer_.fit_transform(texts), y)
        self.tag_vectorizer_ = CountVectorizer(token_pattern=r"\S+", min_df=MIN_TAG_COUNT, binary=True,
                                               dtype=np.float32)
        tag_matrix = self.tag_vectorizer_.fit_transform(tag_tokens(tags)).toarray()
        design = np.hstack([embeddings, numeric, ridge_oof(texts, y)[:, None], tag_matrix])
        self.hgb_ = HistGradientBoostingRegressor(random_state=self.random_state, **self.hgb_params)
        self.hgb_.fit(design, y, sample_weight=sample_weight)
        return self

    def design_matrix(self, embeddings, numeric, texts, tags) -> np.ndarray:
        """Fitur inferensi: skor ridge dari model penuh, bukan out-of-fold."""
        ridge = self.ridge_.predict(self.text_vectorizer_.transform(texts)).astype(np.float32)
        tag_matrix = self.tag_vectorizer_.transform(tag_tokens(tags)).toarray()
        return np.hstack([np.asarray(embeddings, dtype=np.float32), np.asarray(numeric, dtype=np.float32),
                          ridge[:, None], tag_matrix])

    def predict(self, embeddings, numeric, texts, tags) -> np.ndarray:
        return self.hgb_.predict(self.design_matrix(embeddings, numeric, texts, tags))

    @property
    def n_tag_features(self) -> int:
        return len(self.tag_vectorizer_.vocabulary_)
