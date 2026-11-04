#!/usr/bin/env bash
# Menyerahkan keempat job MapReduce (Hadoop Streaming) ke YARN.
#
# Dijalankan dari container `hadoop-client`:
#     docker compose exec hadoop-client bash scripts/mapreduce/run_jobs.sh
#
# Mapper/reducer ditulis dengan Python murni (stdin -> stdout), diserahkan lewat
# `-files` sehingga Hadoop sendiri yang mendistribusikannya ke tiap NodeManager
# (distributed cache) - NodeManager tidak butuh repo ini.
#
# Setelah selesai, cocokkan hasilnya dengan Spark:
#     docker compose exec pipeline python scripts/mapreduce/verify_mapreduce.py
set -euo pipefail

MR=/opt/project/src/mapreduce
REF=/opt/project/Dataset/variety/_ref/stopwords.txt
LOGS=/opt/project/logs/mapreduce
# apache/hadoop:3.3.6 tidak menyetel $HADOOP_HOME (hanya $HADOOP_CONF_DIR) - pakai
# lokasi instalasi tetap, bukan variabel yang tidak ada.
HADOOP_HOME="${HADOOP_HOME:-/opt/hadoop}"
STREAMING=$(ls "$HADOOP_HOME"/share/hadoop/tools/lib/hadoop-streaming-*.jar | head -1)
mkdir -p "$LOGS"
: > "$LOGS/timings.tsv"

# Tepat setelah NameNode start (mis. baru saja `docker compose up`), ia masih
# safe mode sampai menerima cukup laporan blok dari DataNode - biasanya
# beberapa detik, tapi submit job yang jatuh di celah itu gagal dengan
# "Name node is in safe mode" walau HDFS-nya sehat sedetik kemudian.
echo "[mr] menunggu NameNode keluar dari safe mode ..."
for _ in $(seq 1 30); do
  hdfs dfsadmin -safemode get 2>/dev/null | grep -q OFF && break
  sleep 2
done

# run_job <nama> <argumen streaming...>
# Output HDFS dihapus lebih dulu: Hadoop menolak menulis ke direktori yang sudah
# ada, dan itu satu-satunya cara job ini menjadi idempoten.
run_job() {
  local name="$1"; shift
  hdfs dfs -rm -r -f -skipTrash "/novel/output/$name" > /dev/null 2>&1 || true
  echo "[mr] === $name ==="
  local t0=$SECONDS
  # PIPESTATUS harus dibaca SEGERA setelah pipa; `|| true` di ujung akan menimpanya.
  # grep tanpa kecocokan bukan kegagalan job, jadi pipefail dimatikan sementara.
  set +e
  hadoop jar "$STREAMING" -D "mapreduce.job.name=novel-$name" "$@" -output "/novel/output/$name" \
    2>&1 | tee "$LOGS/$name.log" | grep -E "number of splits|map [0-9]+% reduce 100%|completed successfully|ERROR|Exception"
  local status=${PIPESTATUS[0]}
  set -e
  [ "$status" -eq 0 ] || { echo "[mr] $name GAGAL (exit $status) - lihat $LOGS/$name.log" >&2; exit "$status"; }
  printf '%s\t%d\n' "$name" $((SECONDS - t0)) >> "$LOGS/timings.tsv"
  echo "[mr] $name selesai dalam $((SECONDS - t0)) detik"
}

# A. word count - TXT tidak terstruktur. Combiner = reducer (penjumlahan asosiatif).
run_job wordcount \
  -D mapreduce.job.reduces=2 \
  -files "$MR/wordcount_mapper.py,$MR/sum_reducer.py,$REF" \
  -input /novel/unstructured/synopsis \
  -mapper "python3 wordcount_mapper.py" -combiner "python3 sum_reducer.py" -reducer "python3 sum_reducer.py"

# B. frekuensi genre + ko-okurensi - JSONL semi-terstruktur.
run_job genre_stats \
  -D mapreduce.job.reduces=2 \
  -files "$MR/genre_mapper.py,$MR/sum_reducer.py" \
  -input /novel/semistructured/novel_labels.jsonl \
  -mapper "python3 genre_mapper.py" -combiner "python3 sum_reducer.py" -reducer "python3 sum_reducer.py"

# C. statistik per negara asal - CSV terstruktur. Mengeluarkan (cacah, jumlah), bukan rata-rata.
run_job origin_stats \
  -D mapreduce.job.reduces=1 \
  -files "$MR/origin_mapper.py,$MR/sum_reducer.py" \
  -input /novel/structured/novel_metrics.csv \
  -mapper "python3 origin_mapper.py" -combiner "python3 sum_reducer.py" -reducer "python3 sum_reducer.py"

# D. reduce-side join CSV x JSONL, dua tahap berantai.
#    Tahap 1: gabungkan per series_id. Tanpa combiner - join tidak dapat dijumlahkan sebagian.
run_job join_stage1 \
  -D mapreduce.job.reduces=2 \
  -files "$MR/join_mapper.py,$MR/join_reducer.py" \
  -input /novel/structured/novel_metrics.csv -input /novel/semistructured/novel_labels.jsonl \
  -mapper "python3 join_mapper.py" -reducer "python3 join_reducer.py"
#    Tahap 2: jumlahkan (rating, cacah) per genre dari keluaran tahap 1.
run_job genre_rating \
  -D mapreduce.job.reduces=1 \
  -files "$MR/sum_reducer.py" \
  -input /novel/output/join_stage1 \
  -mapper cat -combiner "python3 sum_reducer.py" -reducer "python3 sum_reducer.py"

echo
echo "[mr] semua job selesai. Keluaran: hdfs dfs -ls /novel/output"
hdfs dfs -ls /novel/output
