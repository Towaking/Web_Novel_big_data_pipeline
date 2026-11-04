"""
VELOCITY — ingest inkremental dengan Spark Structured Streaming
Seluruh pipeline (Fase 1-4) sengaja **batch**: Fase 2 memproses ulang seluruh
katalog tiap dijalankan, demi determinisme dan kesederhanaan (lihat docstring
`phase2_clean.py`). Itu keputusan yang benar untuk beban kerja utamanya - tapi
berarti proyek ini tidak punya jalur *velocity* sama sekali, padahal sumbernya
(scraper NovelUpdates) sebenarnya berdatangan bertahap selama berjam-jam.

Modul ini menutup celah itu dengan jalur KEDUA yang independen dari Fase 2:
Structured Streaming yang memonitor direktori *landing* untuk berkas CSV baru
dan mengalirkannya ke tabel curated secara inkremental, satu micro-batch per
berkas yang baru muncul.

**Trigger `availableNow`, bukan `processingTime` tanpa henti.** Untuk skenario
kelas ini (dan supaya bisa diverifikasi otomatis dalam satu perintah, bukan
proses yang harus terus dijaga hidup), query memproses SELURUH berkas yang
tersedia saat itu lalu BERHENTI SENDIRI - tetap mesin streaming sungguhan
(micro-batch, checkpoint, exactly-once), bukan job batch yang disamarkan.
Untuk aliran yang benar-benar tanpa henti, ganti `trigger` ke
`Trigger.ProcessingTime("30 seconds")` dan panggil `query.awaitTermination()`.

**Checkpoint di MinIO** (`s3a://curated-data/streaming/_checkpoint`) yang
mencatat berkas mana yang SUDAH diproses. Jalankan modul ini dua kali dengan
berkas landing yang sama: kedua kali menghasilkan 0 baris baru pada yang
kedua - itulah bukti *exactly-once* dan idempotensi, bukan klaim.

Jalankan:
    docker compose exec pipeline python -m src.phases.streaming_ingest
    docker compose exec pipeline python -m src.phases.streaming_ingest --reset  # hapus checkpoint & tabel
"""

from __future__ import annotations

import argparse
import sys

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from src.common.config import get_settings
from src.common.logging_utils import get_logger, log_section
from src.common.spark_session import spark_session
from src.common.storage import ensure_bucket, get_s3_client, list_objects

logger = get_logger("streaming.ingest")

LANDING_PREFIX = "streaming/landing"
OUTPUT_PREFIX = "streaming/novel_metrics_incremental"
CHECKPOINT_PREFIX = "streaming/_checkpoint"

#: Skema eksplisit wajib untuk file source Structured Streaming: Spark menolak infer-schema karena skema bisa berubah per berkas baru.
LANDING_SCHEMA = StructType(
    [
        StructField("series_id", IntegerType(), True),
        StructField("type_origin", StringType(), True),
        StructField("year", IntegerType(), True),
        StructField("chapter_count", IntegerType(), True),
        StructField("rating_score", DoubleType(), True),
        StructField("rating_votes", IntegerType(), True),
        StructField("weighted_rating", DoubleType(), True),
        StructField("is_rated_reliable", StringType(), True),
        StructField("metadata_tier", StringType(), True),
        StructField("scraped_at_utc", StringType(), True),
    ]
)


def _validate_batch(batch: DataFrame) -> DataFrame:
    """
    Transformasi per micro-batch - versi ringkas normalisasi Fase 2.

    Sengaja sederhana (bukan menyalin ulang `phase2_clean.normalize_sentinels`):
    tujuannya menunjukkan bahwa MICRO-BATCH tetap bisa melalui aturan kualitas
    data yang sama, bukan meniru seluruh Fase 2 sebagai job streaming.
    """
    # Berkas landing berasal dari novel_metrics.csv (data_variety), yang menulis
    # boolean sebagai 1/0 - "true"/"false" tetap diterima untuk sumber lain.
    # Nilai kosong/tak dikenal tetap NULL, bukan dipaksa False.
    flag = F.lower(F.trim(F.col("is_rated_reliable")))
    return (
        batch.filter(F.col("series_id").isNotNull())
        .withColumn(
            "is_rated_reliable",
            F.when(flag.isin("1", "true"), F.lit(True))
            .when(flag.isin("0", "false"), F.lit(False))
            .otherwise(F.lit(None).cast("boolean")),
        )
        .withColumn("ingested_at_utc", F.current_timestamp())
    )


def _count_output(spark: SparkSession, output_uri: str, settings) -> int:
    """Jumlah baris tabel keluaran; 0 bila sink belum pernah menulis batch apa pun."""
    metadata = list_objects(settings.minio.bucket_curated, OUTPUT_PREFIX + "/_spark_metadata/", settings)
    if not metadata:
        return 0
    return spark.read.parquet(output_uri).count()


def build_query(spark: SparkSession, settings=None):
    """Susun (bukan jalankan) query streaming - dipisah agar mudah diuji."""
    settings = settings or get_settings()
    minio = settings.minio
    landing_uri = minio.s3a_uri(minio.bucket_raw, LANDING_PREFIX)
    output_uri = minio.s3a_uri(minio.bucket_curated, OUTPUT_PREFIX)
    checkpoint_uri = minio.s3a_uri(minio.bucket_curated, CHECKPOINT_PREFIX)

    raw_stream = (
        spark.readStream.format("csv")
        .schema(LANDING_SCHEMA)
        .option("header", "true")
        # Satu berkas per micro-batch: pada skala demo ini keterlacakan
        # (berkas mana masuk micro-batch mana, terlihat di log) lebih berharga
        # daripada throughput. Naikkan untuk beban nyata.
        .option("maxFilesPerTrigger", 1)
        .load(landing_uri)
    )
    validated = _validate_batch(raw_stream)

    query = (
        validated.writeStream.format("parquet")
        .option("path", output_uri)
        .option("checkpointLocation", checkpoint_uri)
        .outputMode("append")
        .trigger(availableNow=True)
    )
    return query, output_uri, checkpoint_uri


def reset(settings=None) -> None:
    """Hapus checkpoint & tabel keluaran - untuk mengulang demo dari nol."""
    settings = settings or get_settings()
    client = get_s3_client(settings)
    bucket = settings.minio.bucket_curated
    n = 0
    for prefix in (OUTPUT_PREFIX, CHECKPOINT_PREFIX):
        for obj in list_objects(bucket, prefix, settings):
            client.delete_object(Bucket=bucket, Key=obj["Key"])
            n += 1
    logger.info("Reset: %d objek dihapus dari s3://%s/{%s,%s}", n, bucket, OUTPUT_PREFIX, CHECKPOINT_PREFIX)


def run() -> dict:
    """Jalankan satu putaran ingest inkremental (proses berkas landing baru, lalu berhenti)."""
    log_section(logger, "velocity - structured streaming (ingest inkremental)")
    settings = get_settings()
    ensure_bucket(settings.minio.bucket_raw, settings)
    ensure_bucket(settings.minio.bucket_curated, settings)

    with spark_session("novel-streaming-ingest") as spark:
        query_builder, output_uri, checkpoint_uri = build_query(spark, settings)
        n_before = list_objects(
            settings.minio.bucket_raw, LANDING_PREFIX, settings
        )
        n_landing = sum(1 for o in n_before if o["Key"].endswith(".csv"))
        logger.info("Berkas landing tersedia: %d", n_landing)

        # recentProgress hanya menyimpan 100 entri terakhir secara default, dan
        # maxFilesPerTrigger=1 berarti satu entri per berkas landing.
        spark.conf.set("spark.sql.streaming.numRecentProgressUpdates", str(max(100, n_landing + 10)))
        n_rows_before = _count_output(spark, output_uri, settings)

        query = query_builder.start()
        query.awaitTermination()  # aman: trigger availableNow berhenti sendiri

        progress = query.recentProgress
        total_rows = sum(p.get("numInputRows", 0) for p in progress)
        logger.info(
            "Micro-batch dieksekusi: %d | total baris baru diproses: %d",
            len(progress), total_rows,
        )
        for p in progress:
            logger.info(
                "  batch %s: %d baris dari %s",
                p.get("batchId"), p.get("numInputRows", 0),
                [s.get("description", "?")[:60] for s in p.get("sources", [])],
            )

        n_total_output = _count_output(spark, output_uri, settings)
        if n_total_output - n_rows_before != total_rows:
            logger.warning(
                "Selisih isi tabel (%d) tidak sama dengan numInputRows (%d) - "
                "periksa log micro-batch.", n_total_output - n_rows_before, total_rows,
            )

    logger.info(
        "FASE STREAMING SELESAI | micro-batch=%d | baris baru=%d | total di tabel=%d",
        len(progress), total_rows, n_total_output,
    )
    return {
        "n_landing_files": n_landing,
        "n_microbatches": len(progress),
        "n_new_rows": total_rows,
        "n_total_output_rows": n_total_output,
        "output_uri": output_uri,
        "checkpoint_uri": checkpoint_uri,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Velocity - ingest inkremental via Structured Streaming")
    parser.add_argument("--reset", action="store_true", help="Hapus checkpoint & tabel keluaran lalu keluar")
    args = parser.parse_args()

    try:
        if args.reset:
            reset()
        else:
            run()
    except Exception as exc:  # noqa: BLE001
        logger.error("STREAMING GAGAL: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
