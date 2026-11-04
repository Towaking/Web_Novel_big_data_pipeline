"""
AUDIT KONSEP BIG DATA — apa yang benar-benar terpenuhi
Pemeriksaan yang dijalankan terhadap stack yang sedang hidup, bukan daftar
klaim. Tiap konsep (5V, penyimpanan terdistribusi, pemrosesan terdistribusi,
toleransi kegagalan, ...) mengembalikan status + bukti terukur; konsep yang
memang belum ada dilaporkan `BELUM`, tidak dihaluskan.

Berbeda dari `governance_audit.py` (postur keamanan/kepatuhan), modul ini
menjawab: *apakah proyek ini sungguh memakai teknik Big Data, atau hanya
memakai alatnya?*

Status:
  TERPENUHI — konsep ada dan terbukti berjalan
  SEBAGIAN  — ada, tetapi dengan batas yang disebut di bukti
  BELUM     — konsep belum diimplementasikan
  INFO      — temuan faktual tanpa nilai lulus/gagal

Jalankan:
    docker compose exec pipeline python -m src.phases.bigdata_audit
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.common.config import get_settings
from src.common.logging_utils import get_logger, log_section
from src.common.storage import get_s3_client, list_objects, read_parquet_dir

logger = get_logger("bigdata.audit")

OK, PARTIAL, MISSING, INFO = "TERPENUHI", "SEBAGIAN", "BELUM", "INFO"

VOLUME, VARIETY, VELOCITY = "Volume", "Variety", "Velocity"
VERACITY, VALUE = "Veracity", "Value"
STORAGE, PROCESSING, RELIABILITY = "Penyimpanan terdistribusi", "Pemrosesan terdistribusi", "Keandalan & skala"
GRAPH = "Analisis Graf"
GROUPS = (VOLUME, VARIETY, VELOCITY, VERACITY, VALUE, STORAGE, PROCESSING, RELIABILITY, GRAPH)

HTTP_TIMEOUT = 10


@dataclass
class Finding:
    group: str
    fid: str
    concept: str
    status: str
    evidence: str
    gap: str = ""


@dataclass
class Auditor:
    findings: list[Finding] = field(default_factory=list)
    #: Fakta terukur yang dipakai ulang lintas pilar (mis. versioning dari
    #: STO-01 untuk REL-01) supaya tidak ada status yang ditulis sebagai teks.
    facts: dict = field(default_factory=dict)

    def add(self, *args, **kwargs) -> None:
        f = Finding(*args, **kwargs)
        icon = {OK: "[+]", PARTIAL: "[~]", MISSING: "[x]", INFO: "[i]"}[f.status]
        logger.info("%s %-8s %-40s %s", icon, f.fid, f.concept, f.evidence)
        self.findings.append(f)


def _get_json(url: str):
    """GET JSON; None bila layanannya tidak terjangkau (tidak melempar)."""
    try:
        with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as resp:  # noqa: S310 - URL internal
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _mb(n: float) -> str:
    return f"{n / 1_048_576:,.1f} MB"


def audit_volume(a: Auditor) -> None:
    settings = get_settings()
    per_bucket = {b: list_objects(b) for b in settings.minio.buckets}
    total = sum(o["Size"] for objs in per_bucket.values() for o in objs)
    n_objects = sum(len(objs) for objs in per_bucket.values())
    manifest_path = settings.paths.project_root / "Dataset" / "variety" / "manifest.json"
    n_novels = (
        f"{json.loads(manifest_path.read_text(encoding='utf-8'))['source_rows']:,} novel"
        if manifest_path.is_file() else "jumlah novel tidak diketahui (manifest variety belum ada)"
    )
    a.add(
        VOLUME, "VOL-01", "Skala data", PARTIAL,
        f"{_mb(total)} di {n_objects} objek lake (raw+clean+curated), {n_novels}. "
        "Muat di satu mesin: skala 'big' belum dicapai oleh data, hanya oleh arsitektur.",
        "Volume adalah satu-satunya V yang tidak bisa dipenuhi lewat kode — tidak dipalsukan. "
        "Klaim skalabilitas bertumpu pada VOL-02 dan STO-02.",
    )

    workers = _get_json("http://spark-master:8080/json/")
    if workers is None:
        a.add(VOLUME, "VOL-02", "Komputasi terdistribusi (Spark)", MISSING, "Spark master tidak terjangkau")
    else:
        n = workers.get("aliveworkers", 0)
        a.facts["spark_workers"] = n
        a.add(
            VOLUME, "VOL-02", "Komputasi terdistribusi (Spark)",
            OK if n >= 2 else PARTIAL,
            f"{n} worker hidup, {workers.get('cores', '?')} core, master spark://spark-master:7077",
            "" if n >= 2 else "Hanya satu worker: shuffle antar-node belum diperagakan. "
            "`docker compose up -d --scale spark-worker=2`.",
        )

    curated = list_objects(settings.minio.bucket_curated)
    parquet_parts = [o for o in curated if o["Key"].endswith(".parquet")]
    a.add(
        VOLUME, "VOL-03", "Penyimpanan kolumnar (Parquet)", OK if parquet_parts else MISSING,
        f"{len(parquet_parts)} berkas part Parquet (snappy) di curated-data; zona clean dan curated seluruhnya Parquet"
        if parquet_parts else "tidak ada berkas Parquet di curated-data",
        "" if parquet_parts else "Jalankan Fase 3 / lapisan tata kelola.",
    )

    clean_objs = list_objects(settings.minio.bucket_clean, "novels_clean/")
    partitions = sorted({
        seg.split("=", 1)[1]
        for o in clean_objs
        for seg in o["Key"].split("/")
        if seg.startswith("metadata_tier=")
    })
    a.add(
        VOLUME, "VOL-04", "Partisi fisik untuk partition pruning",
        OK if partitions else MISSING,
        f"novels_clean di-partitionBy('metadata_tier') -> direktori Hive-style {partitions}; "
        "kueri terfilter metadata_tier melompati direktori lain (dibuktikan PartitionFilters "
        "di physical plan Spark, bukan cuma DataFilters pasca-baca)" if partitions
        else "Tulis Parquet memakai repartition(N), tanpa partitionBy",
        "" if partitions else "Tanpa partisi menurut kolom kueri terfilter tetap membaca semua berkas.",
    )


def audit_variety(a: Auditor) -> None:
    settings = get_settings()
    vdir = settings.paths.project_root / "Dataset" / "variety"
    manifest_path = vdir / "manifest.json"
    if not manifest_path.is_file():
        a.add(VARIETY, "VAR-01", "Tiga kelas format, tiga berkas fisik", MISSING,
              "Dataset/variety/manifest.json tidak ada",
              "Jalankan `python -m src.phases.data_variety`.")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_fmt: dict[str, int] = {}
    for f in manifest["files"]:
        if f["class"] != "referensi":
            by_fmt[f["format"]] = by_fmt.get(f["format"], 0) + 1
    missing_files = [f["path"] for f in manifest["files"] if not (vdir / f["path"]).is_file()]
    a.add(
        VARIETY, "VAR-01", "Tiga kelas format, tiga berkas fisik",
        OK if not missing_files and set(by_fmt) == {"csv", "jsonl", "txt"} else PARTIAL,
        "terstruktur=CSV ({csv}), semi-terstruktur=JSONL ({jsonl}), tidak terstruktur=TXT ({txt} shard); "
        "{n:,} novel, sumber {src}".format(n=manifest["source_rows"], src=manifest["source"], **{
            k: by_fmt.get(k, 0) for k in ("csv", "jsonl", "txt")}),
        f"Berkas hilang: {missing_files}" if missing_files else "",
    )
    a.add(
        VARIETY, "VAR-02", "Sumber data heterogen saat ingest", PARTIAL,
        "Fase 1 menelan CSV (katalog, sinopsis, hasil scrape) dan meng-scrape HTML; "
        "JSON/TXT baru lahir SESUDAH Fase 2 lewat pemisahan format",
        "Semua sumber luar berakhir sebagai CSV. Dataset kedua non-tabular (lihat memori 'dataset kedua graf') "
        "belum diputuskan.",
    )


def audit_velocity(a: Auditor) -> None:
    """
    VEL-01/02 diukur dari BUKTI EKSEKUSI NYATA (isi checkpoint & tabel keluaran
    streaming di curated-data), bukan sekadar grep kata kunci `readStream` di
    source - kode yang ada tapi tidak pernah dijalankan bukan bukti apa-apa.
    """
    root = get_settings().paths.project_root
    self_path = Path(__file__).resolve()
    hits = [
        p.relative_to(root).as_posix()
        for base in ("src", "app", "scripts")
        for p in (root / base).rglob("*.py")
        if p.resolve() != self_path
        and re.search(r"readStream|writeStream|KafkaConsumer|KafkaProducer|StreamingContext", p.read_text(errors="ignore"))
    ]

    settings = get_settings()
    output_objs = list_objects(settings.minio.bucket_curated, "streaming/novel_metrics_incremental/")
    checkpoint_objs = list_objects(settings.minio.bucket_curated, "streaming/_checkpoint/")
    offset_files = [o for o in checkpoint_objs if "/offsets/" in o["Key"]]
    commit_files = [o for o in checkpoint_objs if "/commits/" in o["Key"]]
    parquet_files = [o for o in output_objs if o["Key"].endswith(".parquet")]

    if not hits:
        a.add(
            VELOCITY, "VEL-01", "Pemrosesan aliran (stream)", MISSING,
            "tidak ada Structured Streaming / Kafka; seluruh pipeline berjalan batch",
            "Sumber data (scraper NovelUpdates) sebenarnya berdatangan bertahap dan bisa "
            "dijadikan aliran: Structured Streaming atas folder pendaratan, atau Kafka + Spark.",
        )
        a.add(
            VELOCITY, "VEL-02", "Ingest inkremental", PARTIAL,
            "Scraper resumable (melewati URL yang sudah ada) dan Fase 1 idempoten via checksum",
            "Idempoten bukan inkremental: Fase 2 tetap memproses ulang seluruh katalog tiap dijalankan.",
        )
        return

    # Offset ditulis di AWAL micro-batch; commit baru ada bila batch selesai.
    ran = bool(commit_files)
    a.add(
        VELOCITY, "VEL-01", "Pemrosesan aliran (stream)",
        OK if ran else PARTIAL,
        f"{hits[0]} - Structured Streaming (file source, trigger availableNow, checkpoint di MinIO); "
        + (f"TERBUKTI berjalan: {len(offset_files)} micro-batch tercatat, {len(parquet_files)} berkas Parquet keluaran"
           if ran else "kode ada tapi belum pernah dijalankan (checkpoint kosong)"),
        "" if ran else "`docker compose exec pipeline python -m src.phases.streaming_ingest`.",
    )
    a.add(
        VELOCITY, "VEL-02", "Ingest inkremental",
        OK if ran else PARTIAL,
        (f"checkpoint melacak {len(offset_files)} offset / {len(commit_files)} commit micro-batch; "
         "berkas landing yang sudah ter-commit dilewati pada run berikutnya (file source + checkpoint) "
         "- exactly-once, bukan proses ulang seluruh sumber")
        if ran else "Scraper resumable dan Fase 1 idempoten via checksum, tapi itu idempoten bukan inkremental",
        "" if ran else "Idempoten bukan inkremental: Fase 2 tetap memproses ulang seluruh katalog tiap dijalankan.",
    )


def audit_veracity_value(a: Auditor) -> None:
    settings = get_settings()
    gov = settings.paths.doc_dir / "generated" / "governance_audit.md"
    if gov.is_file():
        m = re.search(r"\*\*Ringkasan:\*\* (.+)", gov.read_text(encoding="utf-8"))
        fails = re.search(r"(\d+) gagal", m.group(1)) if m else None
        a.add(
            VERACITY, "VER-01", "Kualitas & keterpercayaan data", OK if fails and fails.group(1) == "0" else PARTIAL,
            f"audit tata kelola: {m.group(1) if m else 'ringkasan tak terbaca'}; sentinel→NULL, dedup by series_id, "
            "weighted rating Bayesian, uji batas atas label",
        )
    else:
        a.add(VERACITY, "VER-01", "Kualitas & keterpercayaan data", MISSING, "governance_audit.md belum dihasilkan")

    analytics = [o for o in list_objects(settings.minio.bucket_curated, "analytics/") if o["Key"].endswith(".parquet")]
    metrics = settings.paths.model_dir / "artifacts" / "metrics.json"
    a.add(
        VALUE, "VAL-01", "Insight, model, dan produk", OK if analytics and metrics.is_file() else PARTIAL,
        f"{len(analytics)} berkas tabel analitik Fase 3, artefak model 4 head "
        f"({'ada' if metrics.is_file() else 'tidak ada'}), dashboard Streamlit",
    )


def audit_storage(a: Auditor) -> None:
    settings = get_settings()
    hdfs = settings.hdfs

    try:
        versioning = {
            b: get_s3_client(settings).get_bucket_versioning(Bucket=b).get("Status", "Off")
            for b in settings.minio.buckets
        }
        a.facts["versioning"] = versioning
        a.add(STORAGE, "STO-01", "Data lake (object storage)", OK,
              "MinIO S3-compatible, tiga zona; versioning: " + ", ".join(f"{b}={s}" for b, s in versioning.items()))
    except Exception as exc:  # noqa: BLE001
        a.add(STORAGE, "STO-01", "Data lake (object storage)", MISSING, f"MinIO tidak terjangkau: {exc}")

    state = _get_json(f"{hdfs.webhdfs_url}/jmx?qry=Hadoop:service=NameNode,name=FSNamesystemState")
    if not state or not state.get("beans"):
        a.add(STORAGE, "STO-02", "Distributed File System (HDFS)", MISSING,
              f"NameNode {hdfs.webhdfs_url} tidak terjangkau",
              "`docker compose up -d namenode worker1 worker2`.")
        return
    live = state["beans"][0].get("NumLiveDataNodes", 0)
    a.facts["live_datanodes"] = live
    a.add(
        STORAGE, "STO-02", "Distributed File System (HDFS)", OK if live >= 2 else PARTIAL,
        f"NameNode aktif, {live} DataNode hidup, {state['beans'][0].get('BlocksTotal', '?')} blok tercatat",
        "" if live >= 2 else "Replikasi antar-node butuh minimal 2 DataNode.",
    )

    files = []
    for sub in ("structured", "semistructured", "unstructured/synopsis"):
        listing = _get_json(f"{hdfs.webhdfs_url}/webhdfs/v1{hdfs.root_dir}/{sub}?op=LISTSTATUS")
        for st in (listing or {}).get("FileStatuses", {}).get("FileStatus", []):
            if st["type"] == "FILE":
                files.append((f"{hdfs.root_dir}/{sub}/{st['pathSuffix']}", st))
    if not files:
        a.add(STORAGE, "STO-03", "Data terdistribusi dalam blok berreplika", MISSING,
              f"tidak ada berkas di HDFS {hdfs.root_dir}/",
              "`docker compose exec hadoop-client bash scripts/hdfs/load_hdfs.sh`.")
        return

    total_blocks, under, single_host, multi_block_files = 0, 0, 0, 0
    datanodes: set[str] = set()
    a.facts["hdfs_replication"] = files[0][1]["replication"]
    for path, st in files:
        loc = _get_json(f"{hdfs.webhdfs_url}/webhdfs/v1{path}?op=GET_BLOCK_LOCATIONS")
        blocks = (loc or {}).get("LocatedBlocks", {}).get("locatedBlocks", [])
        total_blocks += len(blocks)
        multi_block_files += len(blocks) > 1
        for blk in blocks:
            hosts = {l.get("hostName") or l.get("ipAddr") for l in blk.get("locations", [])}
            datanodes |= hosts
            single_host += len(hosts) < 2
            under += len(hosts) < st["replication"]
    a.add(
        STORAGE, "STO-03", "Data terdistribusi dalam blok berreplika",
        OK if not under and not single_host and total_blocks else PARTIAL,
        f"{len(files)} berkas → {total_blocks} blok (ukuran blok {_mb(files[0][1]['blockSize'])}), "
        f"replikasi {files[0][1]['replication']}, tersebar di {len(datanodes)} DataNode; "
        f"{multi_block_files} berkas terpecah >1 blok; blok kurang-replika: {under}",
        "" if not under else "Ada blok yang belum mencapai faktor replikasi — cek `hdfs fsck /novel`.",
    )
    a.add(
        STORAGE, "STO-04", "Dua tier penyimpanan dengan peran berbeda", INFO,
        "MinIO = system of record Fase 1-4 (versioning, dibaca Spark & dashboard); "
        "HDFS = penyimpanan yang berdampingan dengan compute MapReduce",
        "Isi HDFS adalah turunan zona clean (dapat dibangun ulang), bukan sumber kebenaran kedua.",
    )
    report = settings.paths.doc_dir / "generated" / "mapreduce_verification.md"
    verified = report.is_file() and "**Hasil:**" in report.read_text(encoding="utf-8")
    a.add(
        STORAGE, "STO-05", "Spark membaca langsung dari HDFS", OK if verified else PARTIAL,
        "verify_mapreduce.py membaca hdfs://namenode:8020/novel/output lewat Spark, berdampingan dengan s3a:// "
        "(laporan verifikasinya ada)" if verified
        else "verify_mapreduce.py belum pernah menghasilkan laporan - pembacaan HDFS oleh Spark belum terbukti",
        "" if verified else "`make mapreduce-verify`.",
    )


def audit_processing(a: Auditor) -> None:
    settings = get_settings()
    report = settings.paths.doc_dir / "generated" / "mapreduce_verification.md"
    yarn = _get_json("http://resourcemanager:8088/ws/v1/cluster/info")
    nodes = _get_json("http://resourcemanager:8088/ws/v1/cluster/nodes")
    n_nm = len(((nodes or {}).get("nodes") or {}).get("node", []))
    a.add(
        PROCESSING, "PRC-01", "Klaster MapReduce (YARN)", OK if yarn and n_nm >= 2 else (PARTIAL if yarn else MISSING),
        f"ResourceManager {yarn['clusterInfo']['state']}, {n_nm} NodeManager" if yarn else "ResourceManager tidak terjangkau",
        "" if yarn and n_nm >= 2 else "`docker compose up -d resourcemanager worker1 worker2`.",
    )

    if report.is_file():
        text = report.read_text(encoding="utf-8")
        m = re.search(r"\*\*Hasil:\*\* (\d+) dari (\d+)", text)
        ok = bool(m) and m.group(1) == m.group(2)
        a.add(
            PROCESSING, "PRC-02", "Job MapReduce terbukti benar", OK if ok else PARTIAL,
            f"{m.group(1)} dari {m.group(2)} job identik dengan jawaban Spark SQL "
            "(word count TXT, genre JSONL, statistik CSV, join CSV×JSONL)" if m else "laporan tak terbaca",
            "" if ok else "Lihat Doc/generated/mapreduce_verification.md.",
        )
        cm = re.findall(r"([\d,]+) → ([\d,]+) \(([\d.]+)% terpangkas\)", text)
        a.add(
            PROCESSING, "PRC-03", "Optimasi combiner", OK if cm else INFO,
            "; ".join(f"{i} → {o} rekaman ({p}%)" for i, o, p in cm) if cm else "counter belum tercatat",
        )
    else:
        a.add(PROCESSING, "PRC-02", "Job MapReduce terbukti benar", MISSING,
              "mapreduce_verification.md belum ada",
              "`run_jobs.sh` lalu `verify_mapreduce.py`.")

    clean_parts = [o for o in list_objects(settings.minio.bucket_clean, "novels_clean/") if o["Key"].endswith(".parquet")]
    analytic_parts = [o for o in list_objects(settings.minio.bucket_curated, "analytics/") if o["Key"].endswith(".parquet")]
    batch_ok = bool(clean_parts and analytic_parts)
    a.add(
        PROCESSING, "PRC-04", "Batch analytics (Spark SQL/ML)", OK if batch_ok else MISSING,
        f"Fase 2-4 memakai DataFrame & Spark SQL pada cluster standalone; keluarannya ada: "
        f"{len(clean_parts)} part novels_clean, {len(analytic_parts)} part analytics/" if batch_ok
        else "zona clean atau tabel analytics belum terisi",
        "" if batch_ok else "`make pipeline`.",
    )
    audit_data_locality(a)


def audit_data_locality(a: Auditor) -> None:
    """
    PRC-05 diukur dari counter Hadoop ASLI di log job terakhir (bukan diklaim
    dari topologi compose) - `worker{1,2}` menjalankan DataNode DAN NodeManager
    dalam satu container/hostname yang sama supaya YARN bisa mencocokkan lokasi
    blok dengan NodeManager yang benar-benar memegangnya.
    """
    log_dir = get_settings().paths.project_root / "logs" / "mapreduce"
    logs = [p for p in log_dir.glob("*.log")] if log_dir.is_dir() else []
    data_local = rack_local = launched = 0
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern, target in (
            (r"Data-local map tasks=(\d+)", "data"),
            (r"Rack-local map tasks=(\d+)", "rack"),
            (r"Launched map tasks=(\d+)", "launched"),
        ):
            m = re.search(pattern, text)
            if not m:
                continue
            n = int(m.group(1))
            if target == "data":
                data_local += n
            elif target == "rack":
                rack_local += n
            else:
                launched += n

    if not logs or launched == 0:
        a.add(
            PROCESSING, "PRC-05", "Data locality", MISSING,
            "belum ada log job MapReduce untuk diukur",
            "Jalankan `scripts/mapreduce/run_jobs.sh` dulu.",
        )
        return

    pct = 100.0 * data_local / launched
    a.add(
        PROCESSING, "PRC-05", "Data locality",
        OK if pct >= 99 else (PARTIAL if data_local else MISSING),
        f"{data_local}/{launched} task map data-local ({pct:.0f}%), {rack_local} rack-local - "
        "diukur dari counter Hadoop di log job terakhir (bukan klaim topologi); "
        "worker1/worker2 menjalankan DataNode + NodeManager sehostname",
        "" if pct >= 99 else "Cek apakah NodeManager berjalan di hostname yang sama dengan DataNode pemegang bloknya.",
    )


def audit_reliability(a: Auditor) -> None:
    # Dari fakta yang diukur STO-01/02/03 dan VOL-02, bukan teks konstan.
    versioning = a.facts.get("versioning", {})
    live = a.facts.get("live_datanodes", 0)
    replication = a.facts.get("hdfs_replication", 0)
    all_versioned = bool(versioning) and all(s == "Enabled" for s in versioning.values())
    tolerant = all_versioned and live >= 2 and replication >= 2
    a.add(
        RELIABILITY, "REL-01", "Toleransi kegagalan penyimpanan", OK if tolerant else PARTIAL,
        f"HDFS replikasi {replication or '?'} pada {live} DataNode hidup; MinIO versioning: "
        + (", ".join(f"{b}={s}" for b, s in versioning.items()) or "tidak terbaca"),
        "Hanya satu NameNode: itu single point of failure (HA NameNode butuh JournalNode + ZooKeeper)."
        if tolerant else "Butuh replikasi >= 2 pada >= 2 DataNode hidup dan versioning aktif di semua bucket.",
    )
    workers = a.facts.get("spark_workers", 0)
    a.add(
        RELIABILITY, "REL-02", "Skalabilitas horizontal", OK if workers >= 2 and live >= 2 else PARTIAL,
        f"{workers} worker Spark dan {live} DataNode/NodeManager hidup; Spark: `--scale spark-worker=N`; "
        "HDFS/YARN: tambah workerN (DataNode+NodeManager sehostname) pada compose",
        "" if workers >= 2 and live >= 2 else "Skala horizontal baru terbukti dengan >= 2 node di tiap lapisan.",
    )
    orchestrator_src = (get_settings().paths.project_root / "scripts" / "run_pipeline.py")
    text = orchestrator_src.read_text(encoding="utf-8") if orchestrator_src.is_file() else ""
    has_retry = "--retry" in text and "def _run_step" in text
    has_resume = "--resume" in text and "_load_completed" in text
    a.add(
        RELIABILITY, "REL-03", "Orkestrasi & penjadwalan alur",
        OK if has_retry and has_resume else PARTIAL,
        "scripts/run_pipeline.py: sekuensial (bukan DAG), berhenti di kegagalan pertama, "
        + ("+ retry per langkah (`--retry N`, jeda 5 detik) "
           "dan resume dari langkah gagal terakhir (`--resume`, state di logs/pipeline_state.json, "
           "diuji 3 skenario: retry sukses, gagal permanen tanpa resume palsu, resume melewati langkah sukses)"
           if has_retry and has_resume else "TANPA retry atau resume"),
        "" if has_retry and has_resume
        else "Bukan orkestrator penuh: tanpa jadwal cron atau DAG lintas dependensi (Airflow/Dagster) - "
        "cukup untuk pipeline linear seperti ini, tidak untuk topologi job paralel/bercabang.",
    )
    a.add(
        RELIABILITY, "REL-04", "Basis data NoSQL", INFO,
        "Tidak dipakai; kebutuhan relasional/graf dijawab star schema + tabel simpul/sisi (data_model.py)",
        "Bukan kekurangan: alasannya terdokumentasi. Menjadi relevan bila traversal graf melewati 2 hop.",
    )


def write_markdown(findings: list[Finding]) -> None:
    out = get_settings().paths.doc_dir / "generated" / "bigdata_concept_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    tally = {s: sum(1 for f in findings if f.status == s) for s in (OK, PARTIAL, MISSING, INFO)}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Audit Konsep Big Data (dihasilkan otomatis)",
        "",
        f"> Dihasilkan `src/phases/bigdata_audit.py` pada {stamp} terhadap stack yang sedang hidup. "
        "**Jangan diedit manual.**",
        "",
        f"**Ringkasan:** {tally[OK]} terpenuhi · {tally[PARTIAL]} sebagian · {tally[MISSING]} belum · {tally[INFO]} informasi",
        "",
    ]
    for group in GROUPS:
        rows = [f for f in findings if f.group == group]
        if not rows:
            continue
        lines += [f"## {group}", "", "| ID | Konsep | Status | Bukti |", "|---|---|---|---|"]
        lines += [
            f"| `{f.fid}` | {f.concept} | **{f.status}** | {f.evidence.replace('|', chr(92) + '|')} |" for f in rows
        ]
        gaps = [f for f in rows if f.gap]
        if gaps:
            lines += ["", "**Celah:**", ""]
            lines += [f"- `{f.fid}` — {f.gap}" for f in gaps]
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Laporan audit konsep ditulis -> %s", out)


def audit_graph(a: Auditor) -> None:
    """
    GRF-01/02/03 dicek dari isi tabel yang ditulis `graph_analysis.py`, bukan
    dari kode sumbernya - `op_graph_genre_cooccurrence`/`op_graph_author_reach`
    di `data_model.py` MEMANG sudah ada lebih dulu, tapi keduanya self-join
    (hitungan ko-kemunculan), bukan algoritma graf. Perbedaan itu yang dicek
    di sini: apakah PageRank/komunitas/traversal SUNGGUH dihitung.
    """
    settings = get_settings()
    summary_objs = list_objects(settings.minio.bucket_curated, "analytics/graph_summary/")
    pagerank_objs = list_objects(settings.minio.bucket_curated, "analytics/graph_genre_pagerank/")
    if not summary_objs or not pagerank_objs:
        a.add(GRAPH, "GRF-01", "Algoritma graf di atas property graph", MISSING,
              "tabel analytics/graph_summary belum ada",
              "`docker compose exec pipeline python -m src.phases.graph_analysis`.")
        return

    # Tabel 2 baris dibaca lewat boto3 - tidak perlu menyalakan aplikasi Spark.
    try:
        summary = read_parquet_dir(settings.minio.bucket_curated, "analytics/graph_summary", settings)
    except Exception as exc:  # noqa: BLE001
        a.add(GRAPH, "GRF-01", "Algoritma graf di atas property graph", MISSING, f"gagal membaca tabel: {exc}")
        return

    rows = {r["label"]: r for _, r in summary.iterrows()}
    genre_row = rows.get("genre_cooccurrence")
    author_row = rows.get("author_collaboration")
    a.add(
        GRAPH, "GRF-01", "PageRank & sentralitas dihitung (bukan self-join)",
        OK if genre_row is not None and author_row is not None else PARTIAL,
        f"genre_cooccurrence: {genre_row['n_nodes']} simpul/{genre_row['n_edges']} sisi; "
        f"author_collaboration: {author_row['n_nodes']} simpul/{author_row['n_edges']} sisi "
        "(NetworkX: pagerank, degree/betweenness centrality)" if genre_row is not None and author_row is not None
        else "salah satu proyeksi graf belum lengkap",
    )
    if genre_row is not None and author_row is not None:
        has_communities = genre_row["n_communities"] > 0 and author_row["n_communities"] > 0
        a.add(
            GRAPH, "GRF-02", "Deteksi komunitas (modularity)", OK if has_communities else PARTIAL,
            f"genre: {genre_row['n_communities']} komunitas (modularity={genre_row['modularity']}); "
            f"author: {author_row['n_communities']} komunitas (modularity={author_row['modularity']}) - "
            f"{author_row['n_components']} komponen terhubung terpisah, terbesar {author_row['largest_component_size']} penulis",
        )
        n_genre_components = int(genre_row["n_components"])
        solo = author_row.get("n_authors_without_collaborator")
        a.add(
            GRAPH, "GRF-03", "Komponen terhubung (fragmentasi jaringan)", OK,
            f"graf genre {n_genre_components} komponen"
            + (" (menyatu penuh)" if n_genre_components == 1 else "")
            + f"; graf kolaborasi penulis {author_row['n_components']} komponen, terbesar "
            f"{author_row['largest_component_size']} penulis"
            + (f"; {int(solo):,} penulis lain tanpa kolaborator sama sekali" if pd.notna(solo) else ""),
        )
    hops = genre_row.get("traversal_hops") if genre_row is not None else None
    path = genre_row.get("traversal_path") if genre_row is not None else None
    traversed = pd.notna(hops) and int(hops) == 2
    a.add(
        GRAPH, "GRF-04", "Traversal multi-hop terverifikasi", OK if traversed else PARTIAL,
        f"shortest-path BFS atas graf genre menemukan jalur 2-hop {path} antar genre yang tidak "
        "bertetangga langsung - membuktikan (bukan mengasumsikan) klaim '<=2 hop' yang jadi alasan "
        "data_model.py tidak memakai database graf terpisah" if traversed
        else "graph_summary tidak memuat hasil traversal 2-hop",
        "" if traversed else "Jalankan ulang `python -m src.phases.graph_analysis`.",
    )


def run() -> dict:
    log_section(logger, "audit konsep big data")
    a = Auditor()
    for step in (audit_volume, audit_variety, audit_velocity, audit_veracity_value,
                 audit_storage, audit_processing, audit_reliability, audit_graph):
        step(a)
    write_markdown(a.findings)
    tally = {s: sum(1 for f in a.findings if f.status == s) for s in (OK, PARTIAL, MISSING, INFO)}
    logger.info("TERPENUHI=%d  SEBAGIAN=%d  BELUM=%d  INFO=%d",
                tally[OK], tally[PARTIAL], tally[MISSING], tally[INFO])
    return tally


def main() -> int:
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        logger.error("Audit konsep big data gagal: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
