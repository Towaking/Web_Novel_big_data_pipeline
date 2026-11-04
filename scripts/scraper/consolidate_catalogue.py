"""
Satukan hasil scrape NovelUpdates menjadi satu berkas katalog
Keluaran: `Dataset/novel_catalogue.csv`, satu baris per URL seri, 28 kolom
(27 kolom halaman detail + `synopsis`). Inilah satu-satunya sumber yang
diingest Fase 1 dan dibaca Fase 2.

Sumber dan urutan prioritas per URL:
  1. `Dataset/scraped/detail.csv`  - metadata halaman detail (paling baru)
  2. `Dataset/scraped/listing.csv` - judul, genre, sinopsis, status tamat
  3. `Dataset/archive/`            - snapshot katalog sebelumnya; hanya dipakai
     untuk URL yang tidak tercakup scrape (~3 ribu seri)

Yang dilakukan di sini hanya penyeragaman FORMAT antar-sumber (boolean
"Yes"/"True" -> 1/0, dedup per URL). Nilai sentinel (0 bab, "Unknown",
plafon peringkat #2000) dibiarkan apa adanya: membersihkannya tugas Fase 2.

Jalankan ulang setiap kali scraper menulis listing/detail baru:
    python scripts/scraper/consolidate_catalogue.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "Dataset"
OUT = DATASET / "novel_catalogue.csv"

COLUMNS = [
    "series_id", "title", "url", "type", "type_origin", "genres", "tags", "language", "authors",
    "artists", "original_publisher", "english_publisher", "year", "status_coo", "chapter_count",
    "is_completed", "licensed", "completely_translated", "release_frequency", "rating_score",
    "rating_votes", "activity_week_rank", "activity_month_rank", "activity_all_rank",
    "reading_list_count", "reading_list_month_rank", "reading_list_all_rank", "synopsis",
]
BOOL_TEXT = {"yes": 1, "true": 1, "no": 0, "false": 0}


def _bool(series: pd.Series) -> pd.Series:
    """Yes/No, True/False and 0/1 to a nullable 0/1 integer; anything else is empty."""
    text = series.astype("string").str.strip().str.lower()
    mapped = text.map(BOOL_TEXT)
    numeric = pd.to_numeric(text, errors="coerce")
    return mapped.fillna(numeric.where(numeric.isin([0, 1]))).astype("Int64")


def _key(urls: pd.Series) -> pd.Series:
    return urls.astype(str).str.strip().str.rstrip("/").str.lower()


def _synopsis(text: pd.Series) -> pd.Series:
    text = text.astype("string").str.strip()
    return text.where(text.str.len() > 0)


def load_scrape() -> pd.DataFrame:
    listing = pd.read_csv(DATASET / "scraped" / "listing.csv", dtype=str, keep_default_na=False)
    detail = pd.read_csv(DATASET / "scraped" / "detail.csv", dtype=str, keep_default_na=False)

    listing = listing.assign(_key=_key(listing["url"]), synopsis=_synopsis(listing["summary"]))
    # Longest synopsis wins when a URL was written by more than one listing query.
    listing = (listing.assign(_len=listing["synopsis"].str.len().fillna(-1))
               .sort_values("_len", ascending=False).drop_duplicates("_key"))

    votes = pd.to_numeric(detail["rating_votes"], errors="coerce")
    detail = (detail.assign(_key=_key(detail["url"]), _votes=votes.fillna(-1))
              .sort_values("_votes", ascending=False).drop_duplicates("_key"))

    merged = listing[["_key", "url", "title", "genres", "is_completed", "synopsis"]].merge(
        detail.drop(columns=["_votes"]), on="_key", how="left", suffixes=("_l", ""))
    # Detail page values first; the listing fills what a missing detail page left empty.
    for col in ("url", "title", "genres", "is_completed"):
        merged[col] = merged[col].where(merged[col].astype(str).str.len() > 0, merged[f"{col}_l"])
        merged[col] = merged[col].fillna(merged[f"{col}_l"])
    return merged.drop(columns=[c for c in merged.columns if c.endswith("_l")])


def load_archive(known: set[str]) -> pd.DataFrame:
    archive = DATASET / "archive"
    meta = pd.read_csv(archive / "data_novel.csv", dtype=str, keep_default_na=False)
    syn = pd.read_csv(archive / "novel_dataset_scrapped.csv", dtype=str, keep_default_na=False)
    meta = meta.assign(_key=_key(meta["url"]))
    meta = meta[~meta["_key"].isin(known)]
    syn = syn.assign(_key=_key(syn["url"]), synopsis=_synopsis(syn["summary"]))
    syn = (syn.assign(_len=syn["synopsis"].str.len().fillna(-1))
           .sort_values("_len", ascending=False).drop_duplicates("_key"))
    return meta.merge(syn[["_key", "synopsis"]], on="_key", how="left")


def main() -> int:
    scrape = load_scrape()
    archive = load_archive(set(scrape["_key"]))
    catalogue = pd.concat([scrape, archive], ignore_index=True)
    for col in ("is_completed", "licensed", "completely_translated"):
        catalogue[col] = _bool(catalogue[col])
    catalogue = catalogue.reindex(columns=COLUMNS).sort_values("url", kind="stable")

    tmp = OUT.with_suffix(".csv.tmp")
    catalogue.to_csv(tmp, index=False)
    tmp.replace(OUT)
    with_detail = catalogue["series_id"].astype("string").str.len().fillna(0).gt(0).sum()
    print(f"{OUT.relative_to(ROOT)}: {len(catalogue):,} rows "
          f"({len(scrape):,} scraped, {len(archive):,} from the archive; {with_detail:,} with a detail page), "
          f"{OUT.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
