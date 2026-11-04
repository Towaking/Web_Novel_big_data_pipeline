"""
Simulasikan kedatangan bertahap - harness demo/uji untuk `streaming_ingest.py`
Scraper NovelUpdates yang sesungguhnya sudah selesai - datanya sudah ada semua,
bukan sedang mengalir. Skrip ini SENGAJA MENIRU pola kedatangannya: mengambil irisan dari
`Dataset/variety/structured/novel_metrics.csv` (data asli hasil scrape, bukan
data sintetis) dan menuliskannya sebagai berkas landing baru satu per
pemanggilan - persis seperti scraper yang menulis batch baru tiap beberapa
menit ke folder yang dipantau.

**Ini bukan sumber streaming produksi**, ini alat untuk MEMBUKTIKAN bahwa
`streaming_ingest.py` sungguh memproses secara inkremental: jalankan skrip ini
sekali (batch A masuk), jalankan `streaming_ingest.py` (memproses A), jalankan
skrip ini lagi (batch B masuk), jalankan `streaming_ingest.py` lagi (checkpoint
membuktikan HANYA B yang diproses, bukan A+B).

Jalankan:
    docker compose exec pipeline python scripts/streaming/simulate_landing.py --rows 500
    docker compose exec pipeline python scripts/streaming/simulate_landing.py --offset 500 --rows 500
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from datetime import datetime, timezone

from src.common.config import get_settings
from src.common.logging_utils import get_logger
from src.common.storage import ensure_bucket, get_s3_client
from src.phases.streaming_ingest import LANDING_PREFIX, LANDING_SCHEMA

logger = get_logger("streaming.simulate")

SOURCE_CSV = "Dataset/variety/structured/novel_metrics.csv"
LANDING_COLUMNS = [f.name for f in LANDING_SCHEMA.fields if f.name != "scraped_at_utc"]


def read_slice(path, offset: int, rows: int) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        all_rows = list(reader)
    return all_rows[offset : offset + rows]


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulasikan satu batch kedatangan data baru")
    parser.add_argument("--offset", type=int, default=0, help="Baris awal di sumber (0-based)")
    parser.add_argument("--rows", type=int, default=500, help="Jumlah baris per batch")
    args = parser.parse_args()

    settings = get_settings()
    source_path = settings.paths.project_root / SOURCE_CSV
    if not source_path.is_file():
        logger.error("Sumber tidak ditemukan: %s. Jalankan `python -m src.phases.data_variety` dulu.", source_path)
        return 1

    rows = read_slice(source_path, args.offset, args.rows)
    if not rows:
        logger.error("Tidak ada baris pada offset=%d (sumber mungkin sudah habis).", args.offset)
        return 1

    now = datetime.now(timezone.utc)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=LANDING_COLUMNS + ["scraped_at_utc"])
    writer.writeheader()
    for row in rows:
        writer.writerow({**{c: row.get(c, "") for c in LANDING_COLUMNS}, "scraped_at_utc": now.isoformat()})

    key = f"{LANDING_PREFIX}/batch-{now.strftime('%Y%m%dT%H%M%S%f')}.csv"
    ensure_bucket(settings.minio.bucket_raw, settings)
    get_s3_client(settings).put_object(
        Bucket=settings.minio.bucket_raw, Key=key, Body=buf.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )
    logger.info(
        "Batch landing ditulis: s3://%s/%s (%d baris, offset sumber %d-%d)",
        settings.minio.bucket_raw, key, len(rows), args.offset, args.offset + len(rows),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
