#!/usr/bin/env python3
"""
JOB D (tahap 1) - reducer join
Menerima, per `series_id`, record bertanda `R` (rating) dan `G` (genre) dalam
URUTAN SEMBARANG - shuffle hanya menjamin pengelompokan per kunci, bukan urutan
di dalam grup. Karena itu reducer menampung keduanya dulu, baru menyatukan.

Inner join: novel yang hanya punya salah satu sisi (mis. tanpa rating tepercaya)
tidak menghasilkan keluaran. Jumlah yang terbuang dilaporkan lewat counter Hadoop
supaya terlihat di laporan job, bukan hilang diam-diam. Counter dijumlahkan di
memori dan dilaporkan SEKALI di akhir - satu baris stderr per novel akan membanjiri
framework dengan puluhan ribu pesan.

Keluaran, satu baris per pasangan (novel, genre):

    <genre>  <weighted_rating>  1

Tahap 2 (`sum_reducer.py`) menjumlahkan per genre.
"""

import io
import sys


def main():
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")
    dropped = {"join_genre_without_rating": 0, "join_rating_without_genre": 0, "malformed_join_lines": 0}

    def flush(rating, genres):
        if rating is not None and genres:
            for genre in genres.split("|"):
                stdout.write("%s\t%s\t1\n" % (genre, rating))
        elif rating is not None:
            dropped["join_rating_without_genre"] += 1
        else:
            dropped["join_genre_without_rating"] += 1

    current, rating, genres = None, None, None
    for line in stdin:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 3:
            dropped["malformed_join_lines"] += 1
            continue
        key, tag, payload = parts
        if key != current:
            if current is not None:
                flush(rating, genres)
            current, rating, genres = key, None, None
        if tag == "R":
            rating = payload
        elif tag == "G":
            genres = payload
    if current is not None:
        flush(rating, genres)
    stdout.flush()

    for name, count in dropped.items():
        if count:
            sys.stderr.write("reporter:counter:novel,%s,%d\n" % (name, count))


if __name__ == "__main__":
    main()
