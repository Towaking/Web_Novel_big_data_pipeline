"""
Konfigurasi pipeline pemodelan
Seluruh angka yang menentukan perilaku model dikumpulkan di satu berkas, agar
eksperimen cukup mengubah satu tempat dan hasilnya dapat direproduksi.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ARTIFACT_DIR = Path("/opt/project/Model/artifacts")
RANDOM_STATE = 42

#: Worker latih paralel (per genre / per pohon). Container melihat semua core host;
#: OMP_NUM_THREADS=2 membatasi thread BLAS, bukan proses joblib ini.
N_JOBS = int(os.environ.get("PHASE4_N_JOBS", "2"))

#: Genre dengan contoh lebih sedikit dari ini dibuang: klasifikator yang hanya
#: melihat <50 contoh positif selalu memprediksi "bukan" dan mengaburkan macro-F1.
MIN_GENRE_SUPPORT = 50

#: Porsi data uji. Set uji hanya disentuh sekali, di akhir.
TEST_SIZE = 0.2

#: Jumlah fold untuk probabilitas out-of-fold (pemilihan hyperparameter + ambang).
CV_FOLDS = 3


@dataclass(frozen=True)
class FeatureConfig:
    """
    Bobot dan parameter tiap blok fitur teks.

    Ketiga blok teks sengaja tumpang tindih; ablasi menunjukkan ketiganya
    menyumbang (tag saja 0,507 · sinopsis saja 0,455 · gabungan 0,547 makro-F1).
    """

    # Setelah C dan bobot blok dituning, kosakata 40k / 75k / 150k per blok memberi makro-F1
    # OOF 0,6020 / 0,6022 / 0,6027; 75k dipilih (mikro-F1 0,6712 vs 0,6688) di separuh ukuran 150k.
    corpus_min_df: int = 3
    corpus_max_features: int = 75_000
    synopsis_min_df: int = 3
    synopsis_max_features: int = 75_000
    tags_min_df: int = 2
    tags_max_features: int = 15_000
    ngram_range: tuple = (1, 2)
    max_df: float = 0.85
    weights: dict = field(
        default_factory=lambda: {
            # Hasil TPE 40 trial bersama C SVM (OOF 0,6027). Sinopsis sudah termuat di
            # korpus, jadi blok sinopsis sendiri cukup berbobot rendah.
            "corpus": 2.31,
            "synopsis": 0.95,
            "tags": 1.48,
            "meta": 0.48,
        }
    )
    #: Seleksi fitur khusus per label (LIFT, Zhang & Wu 2015). 0 = nonaktif.
    label_specific_k: int = 0


@dataclass(frozen=True)
class ThresholdConfig:
    """Rentang pencarian ambang keputusan per label."""

    #: Di bawah 0,05 model memuntahkan hampir semua genre (presisi runtuh);
    #: di atas 0,80 genre langka tidak pernah lolos.
    low: float = 0.05
    high: float = 0.80
    step: float = 0.025
