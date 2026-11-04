#!/usr/bin/env bash
# Menjalankan keempat job MapReduce TANPA Hadoop, memakai pipa Unix:
#
#     cat input | mapper | sort | reducer
#
# `sort` di tengah adalah fase shuffle & sort versi mini. Berguna untuk
# men-debug mapper/reducer dalam hitungan detik sebelum menyerahkannya ke YARN,
# dan sebagai pembanding: keluaran di sini harus identik dengan hasil di HDFS.
#
# Jalankan dari container pipeline (Python 3 ada di sana):
#     docker compose exec pipeline bash scripts/mapreduce/local_smoke.sh [direktori-keluaran]
set -euo pipefail
export LC_ALL=C  # urutan byte, sama seperti pengurutan kunci Text milik Hadoop

ROOT="${PROJECT_ROOT:-/opt/project}"
DATA="$ROOT/Dataset/variety"
MR="$ROOT/src/mapreduce"
OUT="${1:-/tmp/mr_local}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"  # absolutkan sebelum cd: path relatif tidak berlaku lagi di direktori kerja

# Mapper wordcount membaca stopwords.txt dari direktori kerja (di Hadoop dikirim -files).
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK" && cp "$DATA/_ref/stopwords.txt" .

echo "[A] wordcount     (TXT tidak terstruktur)"
cat "$DATA"/unstructured/synopsis/*.txt \
  | python3 "$MR/wordcount_mapper.py" | sort | python3 "$MR/sum_reducer.py" > "$OUT/wordcount.tsv"

echo "[B] genre_stats   (JSONL semi-terstruktur)"
python3 "$MR/genre_mapper.py" < "$DATA/semistructured/novel_labels.jsonl" \
  | sort | python3 "$MR/sum_reducer.py" > "$OUT/genre_stats.tsv"

echo "[C] origin_stats  (CSV terstruktur)"
python3 "$MR/origin_mapper.py" < "$DATA/structured/novel_metrics.csv" \
  | sort | python3 "$MR/sum_reducer.py" > "$OUT/origin_stats.tsv"

echo "[D] genre_rating  (join CSV x JSONL, 2 tahap)"
{
  mapreduce_map_input_file=/novel/structured/novel_metrics.csv \
    python3 "$MR/join_mapper.py" < "$DATA/structured/novel_metrics.csv"
  mapreduce_map_input_file=/novel/semistructured/novel_labels.jsonl \
    python3 "$MR/join_mapper.py" < "$DATA/semistructured/novel_labels.jsonl"
} | sort | python3 "$MR/join_reducer.py" | sort | python3 "$MR/sum_reducer.py" > "$OUT/genre_rating.tsv"

echo
wc -l "$OUT"/*.tsv
