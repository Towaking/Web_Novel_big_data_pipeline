#!/usr/bin/env python3
"""
JOB A - WORD COUNT atas data TIDAK TERSTRUKTUR (sinopsis, berkas .txt)
Mapper: satu baris `series_id<TAB>title<TAB>synopsis` masuk, satu pasangan
`kata<TAB>1` keluar untuk tiap token yang lolos.

Normalisasi meniru Fase 2 (`phase2_clean.apply_text_processing`) langkah demi
langkah, supaya hasil MapReduce dapat dibandingkan angka-per-angka dengan Spark:

    lower -> [^a-z\\s] jadi spasi -> rapatkan spasi -> pisah spasi
    -> buang token < 3 huruf -> buang stopword

`re.ASCII` disengaja: `\\s` di Java (Spark) hanya mencakup spasi putih ASCII,
sedangkan `\\s` Python secara bawaan mencakup Unicode. Tanpa flag itu dua mesin
memperlakukan NBSP dan sejenisnya berbeda dan hitungan kata melenceng tipis.

Skrip ini tidak memakai apa pun di luar pustaka standar dan kompatibel Python 3.6
karena berjalan di dalam container Hadoop, bukan di image pipeline.

Stopword dibaca dari `stopwords.txt` di direktori kerja (dikirim lewat `-files`).
"""

import io
import re
import sys

MIN_TOKEN_LENGTH = 3
NON_LETTER = re.compile(r"[^a-z\s]", re.ASCII)
WHITESPACE = re.compile(r"\s+", re.ASCII)


def load_stopwords(path="stopwords.txt"):
    with io.open(path, encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def tokenize(text, stopwords):
    text = WHITESPACE.sub(" ", NON_LETTER.sub(" ", text.lower()))
    return [t for t in text.split(" ") if len(t) >= MIN_TOKEN_LENGTH and t not in stopwords]


def main():
    stopwords = load_stopwords()
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")
    for line in stdin:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            # Baris rusak dicatat lewat counter Hadoop, bukan dibuang diam-diam.
            sys.stderr.write("reporter:counter:novel,malformed_text_lines,1\n")
            continue
        for token in tokenize(parts[2], stopwords):
            stdout.write(token + "\t1\n")
    stdout.flush()


if __name__ == "__main__":
    main()
