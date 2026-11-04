"""
Pra-pemrosesan teks
Tahap ini menyiapkan tiga pandangan teks yang dipakai lapisan fitur, sekaligus
menormalkan bentuk kata.

Keputusan penting: **stemming Snowball**. Tanpa itu "cultivation",
"cultivating", dan "cultivator" menjadi tiga fitur terpisah yang masing-masing
terlalu jarang untuk dipelajari pada korpus 3.568 dokumen latih. Stemming
dilakukan di sini (bukan di dalam vectorizer) supaya fungsinya dapat di-pickle
bersama pipeline dan dijalankan ulang secara identik saat inferensi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer

_STEMMER = SnowballStemmer("english")
_STEM_CACHE: dict[str, str] = {}

#: Kolom masukan yang HARUS ada saat training maupun inferensi.
FEATURE_COLUMNS = ("corpus_clean", "synopsis_clean", "tags_clean", "type_origin", "type")


def _stem_word(word: str) -> str:
    """Stemming dengan memo — korpus ini banyak mengulang kata yang sama."""
    cached = _STEM_CACHE.get(word)
    if cached is None:
        cached = _STEMMER.stem(word)
        _STEM_CACHE[word] = cached
    return cached


def stem_texts(texts) -> list[str]:
    """
    Stemming pada level dokumen.

    Fungsi tingkat modul (bukan lambda atau closure) karena ikut tersimpan di
    dalam artefak pipeline dan harus dapat di-pickle.
    """
    series = pd.Series(texts).fillna("").astype(str)
    return [" ".join(_stem_word(token) for token in doc.split()) for doc in series]


def rebuild_clean_text(pdf: pd.DataFrame, stopwords) -> tuple[list[str], list[str]]:
    """
    Bangun ulang `corpus_clean` dan `synopsis_clean` dari teks mentah.

    Identik dengan kolom Fase 2 (diverifikasi pada 39.660 judul) kecuali sensor
    yang dipulihkan, dan memakai tokenizer yang sama dengan jalur inferensi.
    """
    from src.common.text_prep import to_corpus

    stop = set(stopwords)
    title = pdf["title"].fillna("").astype(str)
    synopsis = pdf["synopsis"].fillna("").astype(str)
    tags = pdf["tags"].fillna("").astype(str)
    corpus = [to_corpus(f"{t} {s} {g}", stop) for t, s, g in zip(title, synopsis, tags)]
    return corpus, [to_corpus(s, stop) for s in synopsis]


def build_feature_frame(pdf: pd.DataFrame, stopwords=None) -> pd.DataFrame:
    """
    Bentuk tabel masukan model dari data bersih Fase 2.

    Dengan `stopwords` (Fase 4) teks dibangun ulang lewat `rebuild_clean_text`.
    Tanpanya kolom Fase 2 dipakai apa adanya, supaya angka Fase 5 tetap dapat diulang.

    Kolomnya harus sama persis saat training dan inferensi; daftarnya dikunci
    di `FEATURE_COLUMNS`.
    """
    from src.common.text_prep import tags_text

    if stopwords is not None:
        corpus, synopsis = rebuild_clean_text(pdf, stopwords)
        tags = pdf["tags"].fillna("").map(tags_text)
    else:
        corpus = pdf["corpus_clean"].fillna("")
        synopsis = pdf["synopsis_clean"].fillna("")
        tags = pdf["tags"].fillna("").str.lower().str.replace(",", " ", regex=False)

    return pd.DataFrame(
        {
            # Gabungan judul + sinopsis + tag: menangkap kombinasi lintas-sumber.
            "corpus_clean": list(corpus),
            # Prosa saja: diberi ruang IDF sendiri agar kosakata sinopsis tidak
            # tertimbun frasa tag yang jauh lebih seragam.
            "synopsis_clean": list(synopsis),
            "tags_clean": list(tags),
            "type_origin": pdf["type_origin"].fillna("UNKNOWN"),
            "type": pdf["type"].fillna("UNKNOWN"),
        }
    )


def build_label_matrix(
    genre_lists, min_support: int
) -> tuple[np.ndarray, list[str], "MultiLabelBinarizer"]:
    """
    Ubah daftar genre menjadi matriks biner multi-label.

    Genre langka dibuang lebih dulu: dengan kurang dari `min_support` contoh,
    klasifikatornya selalu memprediksi "bukan" dan hanya menambah derau pada
    macro-F1 tanpa memberi kemampuan prediktif apa pun.
    """
    from sklearn.preprocessing import MultiLabelBinarizer

    rows = [list(g) if g is not None else [] for g in genre_lists]
    counts = pd.Series([g for row in rows for g in row]).value_counts()
    kept = sorted(counts[counts >= min_support].index.tolist())
    filtered = [[g for g in row if g in kept] for row in rows]

    binarizer = MultiLabelBinarizer(classes=kept)
    return binarizer.fit_transform(filtered), kept, binarizer
