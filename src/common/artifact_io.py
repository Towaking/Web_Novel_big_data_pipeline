"""
Simpan dan muat artefak model Fase 4.

Artefak ditulis sebagai pickle protokol 5 tanpa kompresi. `joblib.load` memakai
unpickler Python murni; unpickler C bawaan memuat genre_models.joblib yang sama
dalam 2,0 detik dari bind mount, bukan 3 sampai 4,6 detik, dan berkasnya tetap
terbaca `joblib.load` oleh notebook dan Model/train.py.
"""
from __future__ import annotations

import pickle
from pathlib import Path

#: Byte pertama aliran pickle protokol >= 2 (opcode PROTO).
_PICKLE_PROTO = 0x80


def dump_artifact(obj, path: str | Path) -> None:
    """Tulis satu artefak; objek yang dipakai bersama antar-pipeline tetap disimpan sekali."""
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=5)


def load_artifact(path: str | Path):
    """Pickle biasa lewat unpickler C; berkas joblib terkompresi dari run lama lewat joblib."""
    path = Path(path)
    with open(path, "rb") as handle:
        if handle.read(1) == bytes([_PICKLE_PROTO]):
            handle.seek(0)
            try:
                return pickle.load(handle)
            except Exception:
                # joblib compress=0 juga diawali PROTO, tetapi menyisipkan data array mentah.
                pass
    import joblib

    return joblib.load(path)
