"""
Configuration layer
Satu-satunya sumber kebenaran untuk seluruh setting pipeline.

Prinsip desain (12-factor):
  - Semua nilai berasal dari environment variable (di-inject compose via .env)
  - Ada default yang aman supaya modul tetap bisa diimpor di luar container
  - Objek konfigurasi bersifat immutable (frozen dataclass) agar tidak ada
    fase yang diam-diam memutasi setting fase lain.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:  # dotenv opsional: di dalam container env sudah di-inject compose
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # pragma: no cover
    pass


def _env(key: str, default: str) -> str:
    """Ambil env var; string kosong diperlakukan sebagai 'tidak diset'."""
    value = os.getenv(key, "").strip()
    return value or default


def _env_int(key: str, default: int) -> int:
    raw = _env(key, str(default))
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Environment variable {key!r} harus berupa integer, dapat {raw!r}")


@dataclass(frozen=True)
class MinioConfig:
    """
    Kredensial & endpoint object storage (S3-compatible).

    Kredensialnya berjenjang: `MINIO_ACCESS_KEY`/`MINIO_SECRET_KEY` menang bila
    diset, kalau tidak jatuh ke kredensial root. Jenjang ini yang memungkinkan
    *least privilege* per service tanpa mengubah kode fase mana pun — service
    streamlit dijalankan dengan akun read-only lewat env var, sementara service
    pipeline yang memang harus menulis tetap memakai akun read-write.
    """

    endpoint: str = field(default_factory=lambda: _env("MINIO_ENDPOINT", "https://minio:9000"))
    access_key: str = field(
        default_factory=lambda: _env("MINIO_ACCESS_KEY", _env("MINIO_ROOT_USER", "minioadmin"))
    )
    secret_key: str = field(
        default_factory=lambda: _env(
            "MINIO_SECRET_KEY", _env("MINIO_ROOT_PASSWORD", "minioadmin123")
        )
    )
    bucket_raw: str = field(default_factory=lambda: _env("BUCKET_RAW", "raw-data"))
    bucket_clean: str = field(default_factory=lambda: _env("BUCKET_CLEAN", "clean-data"))
    bucket_curated: str = field(default_factory=lambda: _env("BUCKET_CURATED", "curated-data"))

    @property
    def buckets(self) -> tuple[str, ...]:
        return (self.bucket_raw, self.bucket_clean, self.bucket_curated)

    def s3a_uri(self, bucket: str, key: str = "") -> str:
        """Bangun URI s3a:// yang dipakai Spark untuk baca/tulis ke MinIO."""
        key = key.lstrip("/")
        return f"s3a://{bucket}/{key}" if key else f"s3a://{bucket}"


@dataclass(frozen=True)
class HdfsConfig:
    """
    Endpoint HDFS (distributed file system). MinIO tetap menjadi data lake untuk
    Fase 1-4; HDFS menyimpan dataset hasil pemisahan format yang diproses MapReduce.
    """

    namenode_uri: str = field(default_factory=lambda: _env("HDFS_NAMENODE_URI", "hdfs://namenode:8020"))
    webhdfs_url: str = field(default_factory=lambda: _env("WEBHDFS_URL", "http://namenode:9870"))
    root_dir: str = field(default_factory=lambda: _env("HDFS_ROOT_DIR", "/novel"))


@dataclass(frozen=True)
class SparkConfig:
    """Parameter cluster & tuning Spark."""

    master_url: str = field(default_factory=lambda: _env("SPARK_MASTER_URL", "local[*]"))
    driver_host: str = field(default_factory=lambda: _env("SPARK_DRIVER_HOST", ""))
    driver_memory: str = field(default_factory=lambda: _env("SPARK_DRIVER_MEMORY", "2G"))
    executor_memory: str = field(default_factory=lambda: _env("SPARK_EXECUTOR_MEMORY", "2G"))
    # Dataset ini kecil (~40k baris): 200 shuffle partition default Spark
    # justru bikin overhead task jauh lebih mahal dari komputasinya.
    shuffle_partitions: int = field(default_factory=lambda: _env_int("SPARK_SHUFFLE_PARTITIONS", 8))
    # Kosong = tanpa batas. Diisi untuk driver sekunder (notebook) agar tidak memonopoli core worker.
    cores_max: str = field(default_factory=lambda: _env("SPARK_CORES_MAX", ""))


@dataclass(frozen=True)
class PathConfig:
    """Path lokal di dalam container (bind-mount dari host)."""

    project_root: Path = field(default_factory=lambda: Path(_env("PROJECT_ROOT", "/opt/project")))

    @property
    def source_csv(self) -> Path:
        """Katalog NovelUpdates satu berkas, keluaran scripts/scraper/consolidate_catalogue.py."""
        return self.project_root / _env("SOURCE_CSV", "Dataset/novel_catalogue.csv")

    @property
    def model_dir(self) -> Path:
        return self.project_root / "Model"

    @property
    def doc_dir(self) -> Path:
        return self.project_root / "Doc"


@dataclass(frozen=True)
class Settings:
    """Root configuration object."""

    minio: MinioConfig = field(default_factory=MinioConfig)
    hdfs: HdfsConfig = field(default_factory=HdfsConfig)
    spark: SparkConfig = field(default_factory=SparkConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    raw_object_key: str = field(default_factory=lambda: _env("RAW_OBJECT_KEY", "novel/novel_catalogue.csv"))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper())

    @property
    def raw_sources(self) -> tuple[tuple[Path, str], ...]:
        """Pasangan (file lokal, object key) yang diingest Fase 1."""
        return ((self.paths.source_csv, self.raw_object_key),)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton setting - dibaca sekali per proses."""
    return Settings()
