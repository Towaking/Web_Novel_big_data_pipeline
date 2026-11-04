#!/usr/bin/env bash
# Memuat dataset hasil pemisahan format (Dataset/variety/) ke HDFS, lalu MEMBUKTIKAN
# bahwa blok-bloknya benar-benar tersebar dan berreplika di DataNode berbeda.
#
# Dijalankan dari container `hadoop-client` (punya klien Hadoop + bind-mount repo):
#     docker compose exec hadoop-client bash scripts/hdfs/load_hdfs.sh
#
# Tata letak HDFS - satu direktori per kelas format, sama seperti di disk:
#     /novel/structured/novel_metrics.csv
#     /novel/semistructured/novel_labels.jsonl
#     /novel/unstructured/synopsis/synopsis-000N.txt
#     /novel/ref/stopwords.txt
set -euo pipefail

DATA="${VARIETY_DIR:-/opt/project/Dataset/variety}"
LOGS="/opt/project/logs/hdfs"
ROOT=/novel
REPLICATION="${HDFS_REPLICATION:-2}"
# 4 MB, bukan 128 MB bawaan: seluruh dataset ~60 MB, sehingga dengan blok bawaan
# tiap berkas hanya satu blok dan tidak ada yang terpecah atau tersebar.
# Di produksi, biarkan bawaan.
BLOCK_SIZE="${HDFS_BLOCK_SIZE:-4194304}"

mkdir -p "$LOGS"
[ -f "$DATA/manifest.json" ] || { echo "Jalankan dulu: python -m src.phases.data_variety" >&2; exit 1; }

# Berkas di disk harus sama persis dengan yang dicatat manifest (md5), dan tidak
# boleh ada shard sinopsis di luar manifest - kalau tidak, yang dimuat ke HDFS
# bukan ekspor yang sudah diverifikasi data_variety. Python 3.6: tanpa f-string.
python3 - "$DATA" <<'PY'
import hashlib, json, os, sys
root = sys.argv[1]
manifest = json.load(open(os.path.join(root, "manifest.json")))
listed = set(f["path"] for f in manifest["files"])
bad = []
for f in manifest["files"]:
    path = os.path.join(root, f["path"])
    if not os.path.isfile(path):
        bad.append("hilang: " + f["path"])
        continue
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != f["md5"]:
        bad.append("md5 beda: " + f["path"])
shard_dir = os.path.join(root, "unstructured", "synopsis")
for name in sorted(os.listdir(shard_dir)):
    if name.startswith("synopsis-") and "unstructured/synopsis/" + name not in listed:
        bad.append("di luar manifest: unstructured/synopsis/" + name)
if bad:
    sys.stderr.write("[hdfs] Dataset/variety tidak cocok dengan manifest:\n  " + "\n  ".join(bad) + "\n")
    sys.exit(1)
print("[hdfs] manifest cocok: %d berkas" % len(manifest["files"]))
PY

echo "[hdfs] menunggu NameNode keluar dari safe mode dan >= $REPLICATION DataNode hidup ..."
for _ in $(seq 1 60); do
  live=$(hdfs dfsadmin -report 2>/dev/null | sed -n 's/^Live datanodes (\([0-9]*\)).*/\1/p' | head -1 || true)
  safe=$(hdfs dfsadmin -safemode get 2>/dev/null || true)
  if [ "${live:-0}" -ge "$REPLICATION" ] && [[ "$safe" == *"OFF"* ]]; then break; fi
  sleep 5
done
[ "${live:-0}" -ge "$REPLICATION" ] || { echo "[hdfs] hanya ${live:-0} DataNode hidup, butuh $REPLICATION" >&2; exit 1; }
echo "[hdfs] DataNode hidup: $live"

# Direktori shard dikosongkan dulu: job wordcount membaca SELURUH direktori,
# jadi shard sisa muatan sebelumnya (jumlah shard berbeda) akan ikut terhitung.
hdfs dfs -rm -r -f -skipTrash "$ROOT/unstructured/synopsis" > /dev/null
hdfs dfs -mkdir -p "$ROOT/structured" "$ROOT/semistructured" "$ROOT/unstructured/synopsis" "$ROOT/ref" "$ROOT/output"

put() {  # put <direktori HDFS> <berkas lokal>...
  local dest="$1"; shift
  hdfs dfs -D dfs.blocksize="$BLOCK_SIZE" -D dfs.replication="$REPLICATION" -put -f "$@" "$dest/"
  for f in "$@"; do echo "[hdfs] $(basename "$f") -> $dest/"; done
}
put "$ROOT/structured"            "$DATA/structured/novel_metrics.csv"
put "$ROOT/semistructured"        "$DATA/semistructured/novel_labels.jsonl"
put "$ROOT/unstructured/synopsis" "$DATA"/unstructured/synopsis/synopsis-*.txt
put "$ROOT/ref"                   "$DATA/_ref/stopwords.txt"

echo
echo "[hdfs] === fsck: blok, replika, dan lokasi DataNode ==="
hdfs fsck "$ROOT" -files -blocks -locations > "$LOGS/fsck.txt" 2>&1 || true
grep -E "^(Status|Total (size|files|blocks)|Minimally|Under-replicated|Mis-replicated|Default replication|Average block replication|Corrupt blocks|Missing)" "$LOGS/fsck.txt" || true

hdfs dfsadmin -report > "$LOGS/dfsadmin_report.txt" 2>&1 || true
echo
echo "[hdfs] === kapasitas per DataNode ==="
grep -E "^(Name|Hostname|DFS Used:|Live datanodes)" "$LOGS/dfsadmin_report.txt" || true
echo
echo "[hdfs] laporan lengkap: logs/hdfs/fsck.txt , logs/hdfs/dfsadmin_report.txt"
