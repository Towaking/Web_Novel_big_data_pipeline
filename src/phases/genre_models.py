"""
Kompatibilitas — Head A (genre tagger)
Implementasinya kini berada di paket `Model/pipeline/`, disusun mengikuti
struktur proyek machine learning yang lazim (config → preprocessing →
features → models → evaluate → train).

Modul ini dipertahankan sebagai lapisan tipis karena dua alasan:

  1. `src/phases/phase4_model.py` memanggilnya sebagai bagian dari Fase 4;
  2. artefak model yang sudah tersimpan merujuk nama modul ini di dalam
     pickle-nya, sehingga menghapusnya akan membuat artefak lama gagal dimuat.

Kode baru sebaiknya mengimpor langsung dari `Model.pipeline`.
"""

from Model.pipeline.config import MIN_GENRE_SUPPORT, RANDOM_STATE, TEST_SIZE
from Model.pipeline.evaluate import evaluate, per_label_scores, precision_at_k, tune_thresholds
from Model.pipeline.features import build_feature_union, text_branch as _text_branch
from Model.pipeline.models import (
    DEFAULT_MODELS,
    MODEL_CATALOG,
    PARAM_GRID,
    build_model,
    predict_proba_frame,
)
from Model.pipeline.preprocessing import FEATURE_COLUMNS, stem_texts
from Model.pipeline.train import out_of_fold_probabilities, split_indices, train_all

__all__ = [
    "DEFAULT_MODELS", "MODEL_CATALOG", "PARAM_GRID", "FEATURE_COLUMNS",
    "MIN_GENRE_SUPPORT", "RANDOM_STATE", "TEST_SIZE",
    "build_model", "build_feature_union", "predict_proba_frame", "stem_texts",
    "evaluate", "tune_thresholds", "per_label_scores", "precision_at_k",
    "train_all", "split_indices", "out_of_fold_probabilities", "_text_branch",
]
