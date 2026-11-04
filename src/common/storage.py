"""
Object storage helper
Wrapper tipis di atas boto3 untuk operasi yang TIDAK perlu Spark
(upload file mentah, cek keberadaan objek, listing bucket).

Catatan arsitektur: baca/tulis dataset tetap lewat Spark (`s3a://`).
Modul ini khusus operasi kontrol/metadata yang ringan.
"""

from __future__ import annotations

from pathlib import Path

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

from src.common.config import Settings, get_settings
from src.common.logging_utils import get_logger

logger = get_logger(__name__)


def get_s3_client(settings: Settings | None = None):
    """Buat client S3 yang menunjuk ke MinIO lokal."""
    settings = settings or get_settings()
    minio = settings.minio
    return boto3.client(
        "s3",
        endpoint_url=minio.endpoint,
        aws_access_key_id=minio.access_key,
        aws_secret_access_key=minio.secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        region_name="us-east-1",
    )


def ensure_bucket(bucket: str, settings: Settings | None = None) -> None:
    """Buat bucket bila belum ada (idempotent)."""
    client = get_s3_client(settings)
    try:
        client.head_bucket(Bucket=bucket)
        logger.debug("Bucket '%s' sudah tersedia.", bucket)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchBucket", "403"}:
            client.create_bucket(Bucket=bucket)
            logger.info("Bucket '%s' dibuat.", bucket)
        else:
            raise


def upload_file(
    local_path: Path,
    bucket: str,
    key: str,
    settings: Settings | None = None,
) -> str:
    """Unggah file lokal ke MinIO, kembalikan URI s3a-nya."""
    settings = settings or get_settings()
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError("File sumber tidak ditemukan: " + str(local_path))

    ensure_bucket(bucket, settings)
    try:
        get_s3_client(settings).upload_file(str(local_path), bucket, key)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(
            "Upload ke s3://" + bucket + "/" + key + " gagal: " + str(exc)
        ) from exc

    size_mb = local_path.stat().st_size / 1_048_576
    logger.info("Terunggah: %s -> s3a://%s/%s (%.2f MB)", local_path.name, bucket, key, size_mb)
    return settings.minio.s3a_uri(bucket, key)


def object_exists(bucket: str, key: str, settings: Settings | None = None) -> bool:
    """Cek keberadaan satu objek tanpa mengunduhnya."""
    try:
        get_s3_client(settings).head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def read_parquet_dir(bucket: str, prefix: str, settings: Settings | None = None):
    """
    Baca direktori Parquet hasil tulisan Spark menjadi satu DataFrame pandas.

    Spark menulis satu "file" logis sebagai direktori berisi banyak `part-*`.
    Dashboard tidak memakai Spark (butuh respons milidetik), jadi bagian-bagian
    itu diunduh lewat boto3 lalu digabung di memori. Aman karena tabel yang
    dibaca adalah hasil agregasi Fase 3 — ukurannya ratusan baris, bukan mentah.

    Raises:
        FileNotFoundError: bila prefix tidak berisi file parquet sama sekali.
    """
    import io

    import pandas as pd

    settings = settings or get_settings()
    client = get_s3_client(settings)
    # "/" penutup: tanpa itu prefix 'analytics/top_tags' ikut mengambil
    # 'analytics/top_tags_x/...' dan kedua tabel tergabung diam-diam.
    prefix = prefix.rstrip("/") + "/"
    parts = [
        obj["Key"]
        for obj in list_objects(bucket, prefix, settings)
        if obj["Key"].endswith(".parquet")
    ]
    if not parts:
        raise FileNotFoundError(
            f"Tidak ada file parquet pada s3://{bucket}/{prefix} - "
            "jalankan fase pipeline yang menghasilkannya terlebih dahulu."
        )

    frames = []
    for key in sorted(parts):
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        frames.append(pd.read_parquet(io.BytesIO(body)))
    return pd.concat(frames, ignore_index=True)


def list_objects(bucket: str, prefix: str = "", settings: Settings | None = None) -> list:
    """Listing objek pada satu prefix (dipakai dashboard untuk status pipeline)."""
    client = get_s3_client(settings)
    paginator = client.get_paginator("list_objects_v2")
    results: list = []
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            results.extend(page.get("Contents", []))
    except ClientError as exc:
        logger.warning("Gagal listing s3://%s/%s: %s", bucket, prefix, exc)
    return results
