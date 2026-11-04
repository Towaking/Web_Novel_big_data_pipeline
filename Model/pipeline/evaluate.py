"""
Metrik & penyetelan ambang
Kombinasi metrik mengikuti konvensi dua bidang sekaligus:

* **Klasifikasi genre dari sinopsis** — micro-F1 & macro-F1.
* **Multi-label / extreme multi-label** — Precision@k dan Hamming loss
  (Liu et al., SIGIR 2017).

Penyetelan ambang per label mengikuti Hoang (2018), yang menyebutnya
*probabilistic classification with learned probability threshold*: setiap label
mendapat ambang keputusannya sendiri alih-alih 0,5 seragam.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, hamming_loss

from Model.pipeline.config import ThresholdConfig


def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> float:
    """
    Precision@k — kualitas peringkat, bebas dari pilihan ambang.

    Ini metrik yang paling dekat dengan pengalaman pengguna dashboard, karena
    hasil ditampilkan terurut menurut keyakinan model.
    """
    top_k = np.argsort(-y_prob, axis=1)[:, :k]
    hits = np.take_along_axis(y_true, top_k, axis=1).sum(axis=1)
    # Dokumen dengan kurang dari k label sejati mustahil mencapai 1,0;
    # pembaginya dibatasi agar kasus itu tidak dihukum tidak adil.
    denominator = np.minimum(k, np.maximum(y_true.sum(axis=1), 1))
    return float(np.mean(hits / denominator))


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> dict:
    """Hitung seluruh metrik multi-label pada ambang tertentu."""
    y_pred = (y_prob >= thresholds).astype(int)
    return {
        "f1_micro": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_samples": round(float(f1_score(y_true, y_pred, average="samples", zero_division=0)), 4),
        "hamming_loss": round(float(hamming_loss(y_true, y_pred)), 4),
        "precision_at_1": round(precision_at_k(y_true, y_prob, 1), 4),
        "precision_at_3": round(precision_at_k(y_true, y_prob, 3), 4),
        "precision_at_5": round(precision_at_k(y_true, y_prob, 5), 4),
    }


def tune_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    config: ThresholdConfig | None = None,
) -> np.ndarray:
    """
    Cari ambang optimal untuk setiap genre.

    Ambang tunggal 0,5 mengasumsikan semua genre sama seringnya. Kenyataannya
    Romance ada di 56% katalog sementara Sports di bawah 2% — probabilitas
    Sports nyaris tak pernah menyentuh 0,5 sehingga genre itu tidak pernah
    diprediksi sama sekali. Ini peningkatan tunggal terbesar untuk macro-F1:
    pada Linear SVM, 0,398 -> 0,527.

    PENTING: `y_prob` harus berasal dari prediksi out-of-fold, bukan dari set
    uji. Menyetel ambang lalu melaporkan skor pada data yang sama akan
    melebih-lebihkan performa.
    """
    config = config or ThresholdConfig()
    grid = np.arange(config.low, config.high + 1e-9, config.step)
    thresholds = np.full(len(labels), 0.5, dtype=np.float32)

    for index in range(len(labels)):
        best_score, best_threshold = -1.0, 0.5
        for threshold in grid:
            predicted = (y_prob[:, index] >= threshold).astype(int)
            score = f1_score(y_true[:, index], predicted, zero_division=0)
            if score > best_score:
                best_score, best_threshold = score, float(threshold)
        thresholds[index] = best_threshold

    return thresholds


def blend_probabilities(parts, space: str = "logit") -> np.ndarray:
    """
    Gabungkan probabilitas beberapa model berbobot; dipakai saat latih dan inferensi ensemble.

    "logit" merata-ratakan log-odds (logarithmic pooling, Genest dkk. 1984). Rata-rata probabilitas
    biasa ("prob") membiarkan probabilitas Random Forest yang landai mengacak urutan tiga genre teratas.
    """
    if space == "prob":
        return sum(weight * prob for weight, prob in parts)
    log_odds = 0.0
    for weight, prob in parts:
        clipped = np.clip(prob, 1e-4, 1 - 1e-4)
        log_odds = log_odds + weight * np.log(clipped / (1 - clipped))
    return 1.0 / (1.0 + np.exp(-log_odds))


def per_label_scores(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray, labels: list[str]
) -> dict:
    """F1 tiap genre, diurutkan dari yang terbaik."""
    scores = f1_score(
        y_true, (y_prob >= thresholds).astype(int), average=None, zero_division=0
    )
    return {
        label: round(float(score), 4)
        for label, score in sorted(zip(labels, scores), key=lambda kv: -kv[1])
    }
