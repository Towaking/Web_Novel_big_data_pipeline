#!/usr/bin/env python3
"""
JOB C - STATISTIK PER NEGARA ASAL atas data TERSTRUKTUR (CSV)
Mapper: satu baris CSV masuk; keluar

    <type_origin>  1  <reliable>  <weighted_rating_jika_reliable>  <has_chapters>  <chapter_count>

yaitu *sufficient statistics* (cacah dan jumlah), BUKAN rata-rata. Rata-rata
tidak asosiatif, sehingga combiner yang menghitung rata-rata per map task lalu
reducer merata-ratakannya lagi memberi angka salah tanpa error apa pun. Dengan
(cacah, jumlah) combiner tetap sah, dan rata-rata dihitung sekali di akhir:

    mean = sum / cacah

Reducer + combiner: `sum_reducer.py` (penjumlah vektor).

Header CSV hanya ada di split pertama berkas; map task lain langsung memulai dari
baris data. Karena itu kolom dibaca dari POSISI (kontrak `STRUCTURED_COLUMNS` di
`src/phases/data_variety.py`, dicek `scripts/mapreduce/verify_mapreduce.py`), dan
baris header dikenali dari nilai kolom pertama yang bukan bilangan bulat.
"""

import csv
import io
import sys

# Salinan urutan kolom `data_variety.STRUCTURED_COLUMNS`; disinkronkan oleh verifikasi.
COLUMNS = (
    "series_id", "type", "type_origin", "language", "original_publisher",
    "year", "decade", "chapter_count", "is_completed", "licensed",
    "completely_translated", "rating_score", "rating_votes", "weighted_rating",
    "is_rated_reliable", "reading_list_count", "activity_week_rank",
    "activity_month_rank", "activity_all_rank", "reading_list_month_rank",
    "reading_list_all_rank", "days_per_chapter", "metadata_tier",
)
IDX = {name: i for i, name in enumerate(COLUMNS)}
NULL_KEY = "(null)"


def main():
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace", newline="")
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")
    for row in csv.reader(stdin):
        if len(row) != len(COLUMNS):
            sys.stderr.write("reporter:counter:novel,malformed_csv_rows,1\n")
            continue
        if not row[0].lstrip("-").isdigit():  # baris header
            continue
        origin = row[IDX["type_origin"]] or NULL_KEY
        reliable = row[IDX["is_rated_reliable"]] == "1" and row[IDX["weighted_rating"]] != ""
        rating = row[IDX["weighted_rating"]] if reliable else "0"
        chapters = row[IDX["chapter_count"]]
        stdout.write("\t".join((
            origin, "1", "1" if reliable else "0", rating,
            "1" if chapters else "0", chapters or "0",
        )) + "\n")
    stdout.flush()


if __name__ == "__main__":
    main()
