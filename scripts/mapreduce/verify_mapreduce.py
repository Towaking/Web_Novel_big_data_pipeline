"""
Verifikasi silang MapReduce vs Spark
Empat job MapReduce (Hadoop, dibaca dari HDFS) dibandingkan dengan pertanyaan
yang SAMA yang dijawab Spark dari zona clean (Parquet di MinIO). Dua mesin,
dua jalur baca, dua penyimpanan, satu jawaban yang harus identik.

Lolos berarti dua hal sekaligus: mapper/reducer benar, dan berkas hasil pemisahan
format (`data_variety`) memuat informasi yang sama dengan sumbernya - kalau ekspor
menghilangkan satu baris, hitungan di sini melenceng.

Jalankan (setelah `scripts/mapreduce/run_jobs.sh` selesai):
    docker compose exec pipeline python scripts/mapreduce/verify_mapreduce.py
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.common.config import get_settings
from src.common.logging_utils import get_logger, log_section
from src.common.spark_session import spark_session
from src.mapreduce import join_mapper, origin_mapper
from src.phases.data_variety import STRUCTURED_COLUMNS

logger = get_logger("mapreduce.verify")

FLOAT_TOLERANCE = 1e-6

COUNTERS = (
    "Map input records", "Map output records", "Combine input records",
    "Combine output records", "Reduce input groups", "Reduce input records",
    "Reduce output records", "Reduce shuffle bytes", "Launched map tasks",
    "Launched reduce tasks", "Data-local map tasks", "Rack-local map tasks",
)


def hdfs_uri(path: str) -> str:
    return get_settings().hdfs.namenode_uri.rstrip("/") + path


def read_output(spark: SparkSession, job: str, columns: list[str]) -> DataFrame:
    """Baca keluaran satu job (`part-*` berformat kunci<TAB>nilai...) dari HDFS."""
    raw = spark.read.text(hdfs_uri(f"/novel/output/{job}"))
    parts = F.split(F.col("value"), "\t")
    return raw.select(*[parts.getItem(i).alias(name) for i, name in enumerate(columns)])


def compare(mr: DataFrame, ref: DataFrame, key: str, numeric: list[str]) -> dict:
    """
    Bandingkan keluaran MR dengan jawaban Spark dalam SATU job.

    `diff` = (hanya di MR, hanya di Spark, nilai berbeda) - ketiganya harus 0.
    Keanggotaan sisi ditandai kolom `_mr`/`_ref`, bukan kolom numerik pertama,
    dan NULL di salah satu sisi dihitung sebagai beda: `abs(NULL - x) > tol`
    bernilai NULL dan akan lolos dari filter, sehingga kolom yang hilang dari
    keluaran mapper/reducer dulu terbaca "cocok".
    """
    mr = mr.select(key, F.lit(1).alias("_mr"), *[F.col(c).cast("double").alias(f"mr_{c}") for c in numeric])
    ref = ref.select(key, F.lit(1).alias("_ref"), *[F.col(c).cast("double").alias(f"ref_{c}") for c in numeric])
    joined = mr.join(ref, on=key, how="full_outer")
    both = F.col("_mr").isNotNull() & F.col("_ref").isNotNull()
    differs = F.lit(False)
    for c in numeric:
        a, b = F.col(f"mr_{c}"), F.col(f"ref_{c}")
        bad = (a.isNull() != b.isNull()) | (
            F.abs(a - b) > F.lit(FLOAT_TOLERANCE) * F.greatest(F.lit(1.0), F.abs(b))
        )
        differs = differs | F.coalesce(bad, F.lit(False))
    row = joined.agg(
        F.count("_mr").alias("records"),
        F.count("_ref").alias("ref_records"),
        F.sum((F.col("_ref").isNull()).cast("int")).alias("only_mr"),
        F.sum((F.col("_mr").isNull()).cast("int")).alias("only_ref"),
        F.sum((both & differs).cast("int")).alias("differs"),
    ).first()
    return {
        "records": row["records"],
        "ref_records": row["ref_records"],
        "diff": (int(row["only_mr"] or 0), int(row["only_ref"] or 0), int(row["differs"] or 0)),
    }


def parse_log(job: str) -> dict:
    """Ambil counter & durasi dari log `run_jobs.sh` (bila jobnya sudah pernah dijalankan)."""
    log = get_settings().paths.project_root / "logs" / "mapreduce" / f"{job}.log"
    found: dict = {}
    if not log.is_file():
        return found
    text = log.read_text(encoding="utf-8", errors="replace")
    for name in COUNTERS:
        hit = re.findall(rf"{re.escape(name)}=(\d+)", text)
        if hit:
            found[name] = int(hit[-1])
    return found


def parse_timings() -> dict[str, int]:
    path = get_settings().paths.project_root / "logs" / "mapreduce" / "timings.tsv"
    if not path.is_file():
        return {}
    rows = [line.split("\t") for line in path.read_text().splitlines() if line.strip()]
    return {name: int(seconds) for name, seconds in rows}


def check_column_contract() -> None:
    """Salinan urutan kolom di mapper harus sama dengan sumber kebenarannya."""
    if tuple(origin_mapper.COLUMNS) != STRUCTURED_COLUMNS:
        raise RuntimeError("origin_mapper.COLUMNS menyimpang dari data_variety.STRUCTURED_COLUMNS")
    idx = {name: i for i, name in enumerate(STRUCTURED_COLUMNS)}
    if (join_mapper.WR_INDEX, join_mapper.RELIABLE_INDEX) != (
        idx["weighted_rating"], idx["is_rated_reliable"],
    ):
        raise RuntimeError("indeks kolom join_mapper menyimpang dari STRUCTURED_COLUMNS")
    logger.info("Kontrak kolom mapper = STRUCTURED_COLUMNS (23 kolom) - cocok.")


def run() -> list[dict]:
    log_section(logger, "verifikasi mapreduce vs spark")
    check_column_contract()
    settings = get_settings()
    results: list[dict] = []

    with spark_session("novel-mr-verify") as spark:
        # Hanya kolom yang dipakai keempat pembanding - tanpa ini cache ikut
        # menampung sinopsis dan seluruh salinan korpus teks.
        clean = spark.read.parquet(
            settings.minio.s3a_uri(settings.minio.bucket_clean, "novels_clean")
        ).select(
            "series_id", "synopsis_tokens", "genre_list", "type_origin",
            "is_rated_reliable", "weighted_rating", "chapter_count",
        ).cache()

        mr = read_output(spark, "wordcount", ["word", "n"])
        ref = clean.select(F.explode("synopsis_tokens").alias("word")).groupBy("word").agg(
            F.count("*").alias("n")
        )
        results.append({
            "job": "wordcount", "format": "TXT (tidak terstruktur)",
            **compare(mr, ref, "word", ["n"]),
            "what": "frekuensi kata pada korpus sinopsis",
        })

        mr = read_output(spark, "genre_stats", ["key", "n"])
        exploded = clean.select("series_id", F.explode("genre_list").alias("g"))
        singles = exploded.groupBy("g").agg(F.count("*").alias("n")).select(
            F.concat(F.lit("G:"), "g").alias("key"), "n"
        )
        a, b = exploded.alias("a"), exploded.alias("b")
        pairs = (
            a.join(b, (F.col("a.series_id") == F.col("b.series_id")) & (F.col("a.g") < F.col("b.g")))
            .groupBy("a.g", "b.g").agg(F.count("*").alias("n"))
            .select(F.concat(F.lit("P:"), F.col("a.g"), F.lit("|"), F.col("b.g")).alias("key"), "n")
        )
        ref = singles.unionByName(pairs)
        results.append({
            "job": "genre_stats", "format": "JSONL (semi-terstruktur)",
            **compare(mr, ref, "key", ["n"]),
            "what": "frekuensi genre + ko-okurensi genre berpasangan",
        })

        mr = read_output(spark, "origin_stats",
                         ["origin", "n", "n_reliable", "sum_wr", "n_chapters", "sum_chapters"])
        ref = clean.groupBy(F.coalesce("type_origin", F.lit("(null)")).alias("origin")).agg(
            F.count("*").alias("n"),
            # coalesce ke 0: reducer MR menulis 0, bukan kosong, untuk kelompok tanpa nilai.
            F.coalesce(F.sum(F.col("is_rated_reliable").cast("int")), F.lit(0)).alias("n_reliable"),
            F.sum(F.when(F.col("is_rated_reliable"), F.col("weighted_rating")).otherwise(0.0)).alias("sum_wr"),
            F.count("chapter_count").alias("n_chapters"),
            F.coalesce(F.sum("chapter_count"), F.lit(0)).alias("sum_chapters"),
        )
        results.append({
            "job": "origin_stats", "format": "CSV (terstruktur)",
            **compare(mr, ref, "origin", ["n", "n_reliable", "sum_wr", "n_chapters", "sum_chapters"]),
            "what": "cacah, jumlah rating & jumlah bab per negara asal",
        })

        mr = read_output(spark, "genre_rating", ["genre", "sum_wr", "n"])
        ref = (
            clean.filter(F.col("is_rated_reliable") & (F.size("genre_list") > 0))
            .select("weighted_rating", F.explode("genre_list").alias("genre"))
            .groupBy("genre").agg(F.sum("weighted_rating").alias("sum_wr"), F.count("*").alias("n"))
        )
        results.append({
            "job": "genre_rating", "format": "CSV x JSONL (join lintas format)",
            **compare(mr, ref, "genre", ["sum_wr", "n"]),
            "what": "rata-rata weighted rating per genre (reduce-side join)",
        })

        # Jawaban akhir job D dalam bentuk yang dibaca manusia.
        top = (
            mr.select("genre", (F.col("sum_wr").cast("double") / F.col("n").cast("double")).alias("mean_wr"),
                      F.col("n").cast("int").alias("n"))
            .filter(F.col("n") >= 100).orderBy(F.col("mean_wr").desc()).limit(5).collect()
        )

    for r in results:
        r["ok"] = r["records"] == r["ref_records"] and r["diff"] == (0, 0, 0)
        r["counters"] = parse_log(r["job"])
        logger.info(
            "[%s] %-13s MR=%s Spark=%s | hanya-MR=%d hanya-Spark=%d beda-nilai=%d",
            "LULUS" if r["ok"] else "GAGAL", r["job"], f"{r['records']:,}", f"{r['ref_records']:,}", *r["diff"],
        )
    write_report(results, top)
    return results


def write_report(results: list[dict], top: list) -> None:
    out = get_settings().paths.doc_dir / "generated" / "mapreduce_verification.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    timings = parse_timings()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    passed = sum(1 for r in results if r["ok"])

    lines = [
        "# Verifikasi MapReduce vs Spark (dihasilkan otomatis)",
        "",
        f"> Dihasilkan `scripts/mapreduce/verify_mapreduce.py` pada {stamp}. **Jangan diedit manual.**",
        "",
        f"**Hasil:** {passed} dari {len(results)} job identik dengan jawaban Spark.",
        "",
        "Tiap job MapReduce (Hadoop Streaming, input dari HDFS) dijawab ulang oleh Spark SQL "
        "dari zona clean. Kolom *selisih* = baris yang hanya ada di satu sisi + baris dengan nilai "
        f"berbeda (toleransi relatif {FLOAT_TOLERANCE:g} untuk bilangan pecahan).",
        "",
        "| Job | Format input | Pertanyaan | Baris MR | Baris Spark | Hanya MR | Hanya Spark | Nilai beda | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            "| `{job}` | {format} | {what} | {records:,} | {ref_records:,} | {d0} | {d1} | {d2} | **{st}** |".format(
                d0=r["diff"][0], d1=r["diff"][1], d2=r["diff"][2],
                st="LULUS" if r["ok"] else "GAGAL", **r,
            )
        )

    counter_rows = [r for r in results if r["counters"]]
    if counter_rows:
        lines += ["", "## Counter Hadoop", "",
                  "| Job | Map task | Reduce task | Lokalitas | Map output | Combine in → out | Reduce in | Reduce groups | Shuffle bytes | Durasi (s) |",
                  "|---|---:|---:|---|---:|---|---:|---:|---:|---:|"]
        for r in counter_rows:
            c = r["counters"]
            ci, co = c.get("Combine input records"), c.get("Combine output records")
            combine = f"{ci:,} → {co:,} ({100 * (1 - co / ci):.1f}% terpangkas)" if ci and co is not None else "-"
            data_local, rack_local = c.get("Data-local map tasks", 0), c.get("Rack-local map tasks", 0)
            locality = f"{data_local} data-local, {rack_local} rack-local" if (data_local or rack_local) else "-"
            lines.append(
                f"| `{r['job']}` | {c.get('Launched map tasks', '-')} | {c.get('Launched reduce tasks', '-')} "
                f"| {locality} | {c.get('Map output records', 0):,} | {combine} "
                f"| {c.get('Reduce input records', 0):,} | {c.get('Reduce input groups', 0):,} "
                f"| {c.get('Reduce shuffle bytes', 0):,} | {timings.get(r['job'], '-')} |"
            )
        n_data_local = sum(r["counters"].get("Data-local map tasks", 0) for r in counter_rows)
        n_rack_local = sum(r["counters"].get("Rack-local map tasks", 0) for r in counter_rows)
        n_located = n_data_local + n_rack_local
        if n_located and not n_rack_local:
            locality_note = (
                f"*Lokalitas* {n_data_local}/{n_located} task map data-local: DataNode dan NodeManager "
                "berjalan di container yang sama (`worker1`/`worker2` di `docker-compose.yml`), jadi "
                "YARN menjadwalkan task di node pemegang bloknya."
            )
        elif n_located:
            locality_note = (
                f"*Lokalitas* {n_data_local}/{n_located} task map data-local ({n_rack_local} rack-local): "
                "sebagian task membaca blok lewat jaringan, bukan dari disk node-nya sendiri."
            )
        else:
            locality_note = "*Lokalitas*: counter Data-/Rack-local tidak ditemukan di log job."
        lines += [
            "",
            "*Combine in → out* adalah efek combiner: rekaman yang dijumlahkan di sisi map sebelum "
            "dikirim lewat jaringan. Job `genre_rating` tahap 1 (join) tidak memakai combiner karena "
            "join tidak bisa dijumlahkan sebagian.",
            "",
            locality_note,
        ]

    if top:
        lines += ["", "## Jawaban job `genre_rating` (genre dengan >= 100 novel bernilai)", "",
                  "| Genre | Rata-rata weighted rating | Novel |", "|---|---:|---:|"]
        lines += [f"| {t['genre']} | {t['mean_wr']:.4f} | {t['n']:,} |" for t in top]
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Laporan verifikasi ditulis -> %s", out)


def main() -> int:
    try:
        results = run()
    except Exception as exc:  # noqa: BLE001
        logger.error("Verifikasi MapReduce gagal dijalankan: %s", exc, exc_info=True)
        return 1
    return 0 if all(r["ok"] for r in results) else 2


if __name__ == "__main__":
    sys.exit(main())
