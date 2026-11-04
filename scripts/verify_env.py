"""
Environment smoke test
Dijalankan setelah `docker compose up` untuk memastikan fondasi siap
SEBELUM Fase 1 dieksekusi. Gagal di sini = masalah infrastruktur,
bukan masalah logika pipeline.

Cakupan pemeriksaan:
  1. Konektivitas MinIO + keberadaan ketiga bucket.
  2. File dataset sumber ter-mount ke dalam container.
  3. Spark cluster menerima job (distributed count).
  4. Round-trip Parquet Spark <-> MinIO lewat konektor S3A.

Jalankan:
    docker compose exec pipeline python scripts/verify_env.py
"""

from __future__ import annotations

import sys
import traceback

from src.common.config import get_settings
from src.common.logging_utils import get_logger, log_section
from src.common.storage import get_s3_client, list_objects
from src.common.spark_session import spark_session

logger = get_logger("verify_env")


def check_minio() -> bool:
    """Pastikan MinIO hidup dan seluruh bucket sudah dibuat oleh minio-init."""
    settings = get_settings()
    client = get_s3_client(settings)
    existing = {b["Name"] for b in client.list_buckets().get("Buckets", [])}
    logger.info("Bucket terdeteksi di MinIO: %s", sorted(existing) or "(kosong)")

    missing = [b for b in settings.minio.buckets if b not in existing]
    if missing:
        logger.error("Bucket belum ada: %s -> jalankan ulang service minio-init.", missing)
        return False
    return True


def check_source_dataset() -> bool:
    """Pastikan bind-mount dataset dari host terbaca di dalam container."""
    csv_path = get_settings().paths.source_csv
    if not csv_path.is_file():
        logger.error("Dataset tidak ditemukan pada %s (cek bind-mount di compose).", csv_path)
        return False
    size_mb = csv_path.stat().st_size / 1_048_576
    logger.info("Dataset sumber OK: %s (%.2f MB)", csv_path, size_mb)
    return True


def check_spark_roundtrip() -> bool:
    """Uji eksekusi terdistribusi + tulis/baca Parquet ke MinIO."""
    settings = get_settings()
    probe_uri = settings.minio.s3a_uri(settings.minio.bucket_curated, "_healthcheck/probe")

    with spark_session("healthcheck") as spark:
        # (a) Job terdistribusi sederhana - membuktikan executor bekerja.
        total = spark.range(0, 100_000, numPartitions=4).count()
        logger.info("Distributed count OK: %s baris.", f"{total:,}")

        # (b) Round-trip S3A - membuktikan konektor & kredensial benar.
        df = spark.createDataFrame([(1, "spark"), (2, "minio")], ["id", "komponen"])
        df.write.mode("overwrite").parquet(probe_uri)
        readback = spark.read.parquet(probe_uri)
        rows = readback.count()

        if rows != 2:
            logger.error("Round-trip Parquet gagal: menulis 2 baris, membaca %s.", rows)
            return False

        logger.info("Round-trip Parquet OK di %s", probe_uri)
        readback.show(truncate=False)

    _cleanup_probe()
    return True


def _cleanup_probe() -> None:
    """
    Hapus artefak healthcheck supaya bucket curated tetap bersih.
    Tanpa ini dashboard salah membaca bucket berisi = fase analitik selesai.
    """
    settings = get_settings()
    bucket = settings.minio.bucket_curated
    client = get_s3_client(settings)
    objects = list_objects(bucket, prefix="_healthcheck/", settings=settings)
    if not objects:
        return
    client.delete_objects(
        Bucket=bucket,
        Delete={"Objects": [{"Key": obj["Key"]} for obj in objects]},
    )
    logger.info("Artefak healthcheck dibersihkan (%d objek).", len(objects))


CHECKS = (
    ("Konektivitas MinIO & bucket", check_minio),
    ("Ketersediaan dataset sumber", check_source_dataset),
    ("Spark cluster & konektor S3A", check_spark_roundtrip),
)


def main() -> int:
    log_section(logger, "verifikasi environment big data")
    results: list[tuple[str, bool]] = []

    for label, check in CHECKS:
        logger.info("[CEK] %s ...", label)
        try:
            results.append((label, check()))
        except Exception:  # laporkan semua cek, jangan berhenti di kegagalan pertama
            logger.error("[GAGAL] %s melempar exception:\n%s", label, traceback.format_exc())
            results.append((label, False))

    log_section(logger, "ringkasan")
    for label, ok in results:
        logger.info("%-35s : %s", label, "LULUS" if ok else "GAGAL")

    if all(ok for _, ok in results):
        logger.info("Environment siap. Silakan lanjut ke Fase 1 (Ingestion).")
        return 0

    logger.error("Ada pemeriksaan yang gagal - perbaiki dulu sebelum menjalankan pipeline.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
