#!/usr/bin/env python3
"""
JOB D (tahap 1) - REDUCE-SIDE JOIN lintas format: CSV terstruktur x JSONL semi-terstruktur
Pertanyaan yang dijawab: *rata-rata weighted rating per genre*. Genre ada di
berkas JSONL, rating ada di berkas CSV - tidak ada satu berkas pun yang cukup.

Mapper membaca KEDUA berkas sekaligus dan menandai tiap record dengan asalnya,
dengan `series_id` sebagai kunci join:

    <series_id>  R  <weighted_rating>       dari CSV   (hanya rating tepercaya)
    <series_id>  G  <genre1>|<genre2>|...   dari JSONL

Asal record dikenali dari path berkas input, yang Hadoop Streaming ekspos sebagai
variabel lingkungan `mapreduce_map_input_file`.

Shuffle mengirim semua record ber-`series_id` sama ke reducer yang sama, dan
`join_reducer.py` menyatukannya. Inilah *reduce-side join*: pilihan yang tepat
bila kedua sisi sama-sama besar. Bila salah satu sisi kecil, *map-side join*
(memuat sisi kecil ke memori tiap mapper) menghindari shuffle - tetapi 28 MB
JSONL bukan sisi yang layak dimuat di tiap task, dan tujuannya memang memperagakan
join yang tidak bergantung pada ukuran.
"""

import csv
import io
import json
import os
import sys

WR_INDEX, RELIABLE_INDEX = 13, 14  # posisi di STRUCTURED_COLUMNS


def source_kind():
    path = os.environ.get("mapreduce_map_input_file") or os.environ.get("map_input_file", "")
    if "/semistructured/" in path:
        return "json"
    if "/structured/" in path:
        return "csv"
    sys.stderr.write("join_mapper: sumber tidak dikenali dari path %r\n" % path)
    sys.exit(2)


def main():
    kind = source_kind()
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace", newline="")
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")

    if kind == "csv":
        for row in csv.reader(stdin):
            if len(row) <= RELIABLE_INDEX or not row[0].lstrip("-").isdigit():
                continue  # header atau baris rusak
            if row[RELIABLE_INDEX] == "1" and row[WR_INDEX] != "":
                stdout.write("%s\tR\t%s\n" % (row[0], row[WR_INDEX]))
    else:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                sys.stderr.write("reporter:counter:novel,malformed_json_lines,1\n")
                continue
            genres = sorted(set(record.get("genres") or []))
            if genres:
                stdout.write("%s\tG\t%s\n" % (record["series_id"], "|".join(genres)))
    stdout.flush()


if __name__ == "__main__":
    main()
