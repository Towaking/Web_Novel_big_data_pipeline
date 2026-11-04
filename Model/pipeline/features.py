"""
Rekayasa fitur
Menyusun ruang fitur dari tiga pandangan teks + metadata kategorikal.

Ketiga blok teks tumpang tindih dan itu disengaja — hasil ablasi menunjukkan
masing-masing menyumbang:

    tag saja        makro-F1 0,507
    sinopsis saja   makro-F1 0,455
    ketiganya       makro-F1 0,547

Temuan yang tidak intuitif: menambahkan blok tag **utuh** (indikator biner per
tag) justru menurunkan skor menjadi 0,491. Sebabnya `corpus_clean` sudah memuat
tag di dalamnya, sehingga blok tambahan hanya menggandakan informasi yang sama
dan mengencerkan sinyal sinopsis. Yang berhasil adalah kebalikannya: memberi
sinopsis ruang IDF-nya sendiri.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from Model.pipeline.config import FeatureConfig
from Model.pipeline.preprocessing import stem_texts


def text_branch(min_df: int, max_features: int, config: FeatureConfig) -> Pipeline:
    """Satu cabang teks: stemming -> TF-IDF n-gram.

    `sublinear_tf` meredam dominasi kata yang berulang di satu dokumen — penting
    di sini karena beberapa tag muncul berkali-kali dalam korpus gabungan.
    """
    return Pipeline(
        [
            ("stem", FunctionTransformer(stem_texts, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=config.ngram_range,
                    min_df=min_df,
                    max_df=config.max_df,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def build_feature_union(config: FeatureConfig | None = None) -> ColumnTransformer:
    """Gabungkan seluruh blok fitur menjadi satu matriks sparse."""
    config = config or FeatureConfig()
    transformers = [
        ("corpus", text_branch(config.corpus_min_df, config.corpus_max_features, config), "corpus_clean"),
        ("synopsis", text_branch(config.synopsis_min_df, config.synopsis_max_features, config), "synopsis_clean"),
        ("tags", text_branch(config.tags_min_df, config.tags_max_features, config), "tags_clean"),
        (
            # Asal negara sangat menentukan genre: Xianxia praktis hanya ada
            # pada novel Tiongkok. Informasi ini tersedia saat prediksi,
            # jadi tidak memakainya berarti membuang sinyal gratis.
            "meta",
            OneHotEncoder(handle_unknown="ignore", min_frequency=20),
            ["type_origin", "type"],
        ),
    ]

    return ColumnTransformer(
        transformers=transformers,
        transformer_weights=dict(config.weights),
        sparse_threshold=1.0,
    )


class MultiLabelChi2Selector(BaseEstimator, TransformerMixin):
    """
    Pilih `k` fitur dengan skor chi-kuadrat tertinggi di genre mana pun.

    Untuk Random Forest: sqrt(p) fitur acak dari ratusan ribu kolom jarang hampir
    selalu kolom nol. Skornya maksimum lintas genre agar penanda genre langka ikut.
    """

    def __init__(self, k: int = 3000):
        self.k = k

    def fit(self, X, y):
        labels = np.asarray(y)
        if labels.ndim == 1:
            labels = labels[:, None]
        scores = np.zeros(X.shape[1])
        for column in range(labels.shape[1]):
            chi, _ = chi2(X, labels[:, column])
            scores = np.maximum(scores, np.nan_to_num(chi))
        keep = min(self.k, X.shape[1])
        self.support_ = np.sort(np.argsort(-scores, kind="mergesort")[:keep])
        return self

    def transform(self, X):
        return X[:, self.support_]


def label_specific_selector(k: int):
    """
    Seleksi fitur khusus per label (semangat LIFT, Zhang & Wu 2015).

    LIFT membangun ruang fitur berbeda untuk tiap label karena label yang
    berbeda ditentukan oleh karakteristik yang berbeda. Versi ringan yang
    dipakai di sini: seleksi chi-kuadrat per label di dalam One-vs-Rest,
    sehingga tiap klasifikator biner hanya melihat fitur yang relevan baginya.

    Mengembalikan None bila dinonaktifkan (k <= 0).
    """
    if k <= 0:
        return None
    return SelectKBest(chi2, k=k)
