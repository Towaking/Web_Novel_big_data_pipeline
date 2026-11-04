"""
FASE 1 — COLLECT (Ingestion)
Memindahkan `Dataset/novel_catalogue.csv` dari filesystem lokal ke landing zone
data lake: `s3a://raw-data/novel/novel_catalogue.csv`.

Prinsip yang dipegang di layer ingestion:

  1. **Raw tetap raw.** Tidak ada transformasi apa pun di fase ini. Bucket
     `raw-data` adalah salinan bit-per-bit sumber, sehingga seluruh pipeline
     hilir selalu bisa di-replay dari titik yang sama.
  2. **Idempoten.** Ingestion yang diulang tidak menggandakan atau merusak
     data. Checksum MD5 dibandingkan lebih dulu; upload dilewati bila objek
     di lake sudah identik (kecuali dipaksa dengan --force).
  3. **Dapat diaudit.** Setiap ingestion menulis manifest berisi checksum,
     ukuran, jumlah baris, dan timestamp — bukti asal-usul data (provenance)
     saat hasil analitik dipertanyakan.

Jalankan:
    docker compose exec pipeline python -m src.phases.phase1_collect [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import get_settings
from src.common.logging_utils import get_logger, log_section
from src.common.storage import get_s3_client, object_exists, upload_file

logger = get_logger("phase1.collect")

MANIFEST_KEY_SUFFIX = "_manifest.json"


def compute_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Checksum streaming supaya file besar tidak perlu masuk memori sekaligus."""
    digest = hashlib.md5()  # noqa: S324 - dipakai untuk integritas, bukan kriptografi
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    """
    Hitung baris data (tanpa header).

    Memakai csv.reader, bukan sekadar menghitung newline, karena kolom `tags`
    dan `title` berisi teks berkutip yang bisa memuat newline di dalamnya.
    """
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def read_remote_manifest(bucket: str, key: str) -> dict | None:
    """Ambil manifest ingestion sebelumnya, bila ada."""
    if not object_exists(bucket, key):
        return None
    try:
        body = get_s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    except (ValueError, KeyError) as exc:
        logger.warning("Manifest lama tidak terbaca (%s) - dianggap tidak ada.", exc)
        return None


def build_manifest(source: Path, checksum: str, rows: int, object_uri: str) -> dict:
    """Metadata provenance yang menyertai setiap ingestion."""
    return {
        "source_file": source.name,
        "object_uri": object_uri,
        "size_bytes": source.stat().st_size,
        "md5": checksum,
        "row_count": rows,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline_phase": "1-collect",
    }


def ingest_file(source: Path, key: str, force: bool = False) -> dict:
    """
    Ingest satu file ke landing zone.

    Returns:
        Manifest ingestion file tersebut.

    Raises:
        FileNotFoundError: dataset sumber tidak ada di bind-mount.
        RuntimeError: upload gagal atau verifikasi pasca-upload tidak cocok.
    """
    settings = get_settings()
    bucket = settings.minio.bucket_raw
    # Manifest per file, dinamai mengikuti nama objeknya, supaya kedua sumber
    # punya jejak provenance masing-masing dan bisa diingest secara terpisah.
    manifest_key = f"{Path(key).parent.as_posix()}/{Path(key).stem}{MANIFEST_KEY_SUFFIX}"

    if not source.is_file():
        raise FileNotFoundError(
            f"Dataset sumber tidak ditemukan: {source}. "
            "Periksa bind-mount ./:/opt/project pada docker-compose.yml."
        )

    checksum = compute_md5(source)
    size_mb = source.stat().st_size / 1_048_576
    logger.info("Sumber   : %s (%.2f MB)", source, size_mb)
    logger.info("Checksum : md5=%s", checksum)

    previous = read_remote_manifest(bucket, manifest_key)
    if previous and previous.get("md5") == checksum and not force:
        logger.info(
            "Objek identik sudah ada di lake (diingest %s) - upload dilewati. "
            "Pakai --force untuk menimpa.",
            previous.get("ingested_at_utc", "?"),
        )
        return previous

    if previous and previous.get("md5") != checksum:
        logger.warning(
            "Checksum berubah dari ingestion sebelumnya (%s -> %s). "
            "Data sumber diperbarui, lake akan ditimpa.",
            previous.get("md5", "?")[:8],
            checksum[:8],
        )

    # Dihitung setelah cek idempotensi: parse CSV multi-baris 40+ MB hanya
    # berguna untuk manifest baru, bukan untuk upload yang dilewati.
    rows = count_csv_rows(source)
    logger.info("Baris data: %s", f"{rows:,}")

    object_uri = upload_file(source, bucket, key, settings)

    # Verifikasi pasca-upload: upload "sukses" yang terpotong membuat fase hilir berjalan mulus di atas data tidak lengkap.
    head = get_s3_client(settings).head_object(Bucket=bucket, Key=key)
    remote_size = head["ContentLength"]
    if remote_size != source.stat().st_size:
        raise RuntimeError(
            f"Verifikasi gagal: ukuran lokal {source.stat().st_size} B "
            f"vs objek di lake {remote_size} B."
        )
    logger.info("Verifikasi ukuran objek cocok (%s byte).", f"{remote_size:,}")

    manifest = build_manifest(source, checksum, rows, object_uri)
    get_s3_client(settings).put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=json.dumps(manifest, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info("Manifest ditulis ke s3a://%s/%s", bucket, manifest_key)
    return manifest


def run(force: bool = False) -> dict:
    """
    Ingest katalog sumber apa adanya ke zona raw. Landing zone tetap cerminan
    sumber; pembersihan seluruhnya urusan Fase 2.
    """
    log_section(logger, "fase 1 - collect (ingestion ke minio)")

    settings = get_settings()
    manifests: dict[str, dict] = {}

    for source, key in settings.raw_sources:
        logger.info("--- Ingest: %s ---", source.name)
        manifests[source.name] = ingest_file(source, key, force=force)

    total_rows = sum(m.get("row_count", 0) for m in manifests.values())
    logger.info("FASE 1 SELESAI - %d file, %s baris total di zona raw.",
                len(manifests), f"{total_rows:,}")
    return manifests


def main() -> int:
    parser = argparse.ArgumentParser(description="Fase 1 - ingestion CSV ke MinIO")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Timpa objek di lake walaupun checksum-nya sama.",
    )
    args = parser.parse_args()

    try:
        run(force=args.force)
    except Exception as exc:  # noqa: BLE001 - entrypoint: laporkan lalu keluar rapi
        logger.error("FASE 1 GAGAL: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
