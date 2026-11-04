"""
VARIETY — memisahkan dataset per kelas format
Konsep *Variety* pada Big Data baru terpenuhi kalau ketiga kelas format benar-benar
tersimpan sebagai **berkas berbeda dengan format yang cocok untuk sifatnya**, bukan
sekadar diberi label di dalam satu CSV. `data_catalog.py` sudah mengklasifikasikan
tiap elemen; modul ini mewujudkan klasifikasi itu secara fisik.

| Kelas            | Berkas                                    | Format | Kenapa format ini                              |
|------------------|-------------------------------------------|--------|------------------------------------------------|
| Terstruktur      | `structured/novel_metrics.csv`            | CSV    | kolom atomik bertipe tetap, siap `GROUP BY`    |
| Semi-terstruktur | `semistructured/novel_labels.jsonl`       | JSONL  | array berpanjang bervariasi + objek bersarang  |
| Tidak terstruktur| `unstructured/synopsis/synopsis-NNNN.txt` | TXT    | prosa bebas, diproses per token bukan per kolom|

Keputusan desain yang tidak kelihatan dari hasilnya:

  * **JSONL, bukan satu berkas JSON besar.** Satu objek per baris membuat berkas bisa
    dipecah di batas baris (HDFS block / input split MapReduce) dan dibaca streaming.
    Array JSON tunggal harus di-parse utuh sebelum satu record pun terbaca.
  * **TXT dishard, bukan satu berkas per novel.** 39 ribu berkas kecil adalah *small
    files problem* klasik: setiap berkas memakan satu entri memori NameNode dan satu
    map task MapReduce, sehingga overhead penjadwalan mengalahkan komputasinya.
    Sharding jadi beberapa berkas ~9 MB berisi satu novel per baris
    (`series_id<TAB>title<TAB>synopsis`, spasi putih diratakan) menjaga sifat
    line-oriented yang dibutuhkan mapper.
  * **Sumbernya zona clean, bukan CSV mentah.** Ketiga berkas turunan dari tabel
    Parquet yang sama, jadi jumlah baris ketiganya bisa dicocokkan satu sama lain
    dan terhadap sumbernya (lihat `verify`). Teks yang diekspor adalah `synopsis`
    ASLI, bukan `synopsis_clean`: tokenisasi dan stopword removal sengaja dikerjakan
    ulang oleh mapper MapReduce, dan hasilnya dibandingkan dengan Spark.

Keluaran ditulis ke `Dataset/variety/` (bind-mount, terlihat di host) beserta
`manifest.json` berisi checksum tiap berkas. Penyimpanan ke HDFS dilakukan langkah
terpisah (`scripts/hdfs/load_hdfs.sh`) karena HDFS dioperasikan dari container Hadoop.

Jalankan:
    docker compose exec pipeline python -m src.phases.data_variety
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import functions as F

from src.common.config import get_settings
from src.common.logging_utils import get_logger, log_section
from src.common.spark_session import spark_session
from src.phases.phase2_clean import DOMAIN_STOPWORDS

logger = get_logger("data.variety")

#: Kolom atomik untuk berkas terstruktur. Urutannya adalah kontrak dengan mapper
#: MapReduce (`src/mapreduce/`), yang membaca kolom berdasarkan nama header.
STRUCTURED_COLUMNS = (
    "series_id", "type", "type_origin", "language", "original_publisher",
    "year", "decade", "chapter_count", "is_completed", "licensed",
    "completely_translated", "rating_score", "rating_votes", "weighted_rating",
    "is_rated_reliable", "reading_list_count", "activity_week_rank",
    "activity_month_rank", "activity_all_rank", "reading_list_month_rank",
    "reading_list_all_rank", "days_per_chapter", "metadata_tier",
)

#: Jumlah shard berkas teks. 4 shard x ~9 MB > blok HDFS yang diperkecil di
#: `load_hdfs.sh`, sehingga tiap shard terpecah jadi beberapa blok berreplika.
DEFAULT_TEXT_SHARDS = 4

SOURCE_COLUMNS = (
    *STRUCTURED_COLUMNS, "title", "url", "genre_list", "tag_list", "authors",
    "status_coo", "release_frequency", "synopsis",
)


def variety_dir() -> Path:
    return get_settings().paths.project_root / "Dataset" / "variety"


def _flat(text: str | None) -> str:
    """Ratakan seluruh spasi putih (newline, tab) supaya satu novel = satu baris."""
    return " ".join(text.split()) if text else ""


def _cell(value):
    """Nilai CSV: NULL -> sel kosong, boolean -> 0/1 (seperti sumber aslinya)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    return value


def _label_record(row) -> dict:
    """Satu objek JSON per novel: array dan objek bersarang, bukan kolom datar."""
    authors = [a.strip() for a in (row["authors"] or "").split(",") if a.strip()]
    # Baris tier 'listing' belum pernah diambil halaman detailnya: status tamat
    # sebenarnya TIDAK DIKETAHUI, jadi jangan diekspor sebagai `false`.
    completed = row["is_completed"] if row["metadata_tier"] == "full" else None
    return {
        "series_id": row["series_id"],
        "title": row["title"],
        "url": row["url"],
        "genres": list(row["genre_list"] or []),
        "tags": list(row["tag_list"] or []),
        "authors": authors,
        "status": {
            "raw": row["status_coo"],
            "chapter_count": row["chapter_count"],
            "is_completed": None if completed is None else bool(completed),
        },
        "release": {
            "raw": row["release_frequency"],
            "days_per_chapter": row["days_per_chapter"],
        },
    }


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - integritas, bukan kriptografi
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_stopwords(out_dir: Path) -> Path:
    """
    Tulis daftar stopword yang dipakai Fase 2 supaya mapper memakai daftar IDENTIK.

    Daftar ini diambil dari Spark ML + `DOMAIN_STOPWORDS`, bukan disalin tangan:
    salinan tangan akan diam-diam melenceng saat Fase 2 diubah, dan hasil
    MapReduce berhenti cocok dengan Spark tanpa ada yang tahu kenapa.
    """
    words = sorted(
        {w.lower() for w in StopWordsRemover.loadDefaultStopWords("english")}
        | {w.lower() for w in DOMAIN_STOPWORDS}
    )
    path = out_dir / "_ref" / "stopwords.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(words) + "\n", encoding="utf-8")
    return path


def export(shards: int = DEFAULT_TEXT_SHARDS) -> dict:
    """Ekspor zona clean menjadi tiga berkas berformat berbeda + manifest."""
    settings = get_settings()
    out_dir = variety_dir()
    struct_path = out_dir / "structured" / "novel_metrics.csv"
    semi_path = out_dir / "semistructured" / "novel_labels.jsonl"
    text_dir = out_dir / "unstructured" / "synopsis"
    # manifest.json dihapus dulu dan ditulis `run()` setelah verifikasi baca-ulang lulus, agar ekspor yang gagal
    # tidak meninggalkan manifest lama (load_hdfs.sh memakainya sebagai tanda siap).
    for old in (out_dir / "manifest.json", struct_path, semi_path, *text_dir.glob("synopsis-*.txt")):
        old.unlink(missing_ok=True)  # jangan biarkan shard sisa dari ekspor lebih besar
    for directory in (struct_path.parent, semi_path.parent, text_dir):
        directory.mkdir(parents=True, exist_ok=True)

    with spark_session("novel-variety-export") as spark:
        clean = spark.read.parquet(
            settings.minio.s3a_uri(settings.minio.bucket_clean, "novels_clean")
        )
        counts = clean.agg(
            F.count(F.lit(1)).alias("n"),
            F.count(F.when(F.length(F.trim(F.col("synopsis"))) > 0, 1)).alias("text"),
        ).first()
        n_clean, n_text = counts["n"], counts["text"]
        logger.info("Zona clean: %s baris, %s bersinopsis.", f"{n_clean:,}", f"{n_text:,}")

        # Urut by series_id: ekspor deterministik, dan shard teks berisi rentang
        # id yang berdekatan. `toLocalIterator` mengalirkan satu partisi sekali
        # waktu, jadi driver tidak pernah menampung seluruh korpus.
        rows = clean.select(*SOURCE_COLUMNS).orderBy("series_id").toLocalIterator()

        per_shard = -(-n_text // shards)
        text_files: list[Path] = []
        text_out = None
        n_struct = n_semi = n_written_text = 0

        with struct_path.open("w", encoding="utf-8", newline="") as f_struct, \
             semi_path.open("w", encoding="utf-8", newline="\n") as f_semi:
            writer = csv.writer(f_struct, lineterminator="\n")
            writer.writerow(STRUCTURED_COLUMNS)
            try:
                for row in rows:
                    writer.writerow([_cell(row[c]) for c in STRUCTURED_COLUMNS])
                    n_struct += 1

                    f_semi.write(json.dumps(_label_record(row), ensure_ascii=False) + "\n")
                    n_semi += 1

                    synopsis = _flat(row["synopsis"])
                    if not synopsis:
                        continue
                    if n_written_text % per_shard == 0:
                        if text_out:
                            text_out.close()
                        text_files.append(text_dir / f"synopsis-{len(text_files):04d}.txt")
                        text_out = text_files[-1].open("w", encoding="utf-8", newline="\n")
                    text_out.write(f"{row['series_id']}\t{_flat(row['title'])}\t{synopsis}\n")
                    n_written_text += 1
            finally:
                if text_out:
                    text_out.close()

    stopwords = write_stopwords(out_dir)

    outputs = [
        ("terstruktur", "csv", struct_path, n_struct),
        ("semi-terstruktur", "jsonl", semi_path, n_semi),
        *[("tidak-terstruktur", "txt", p, None) for p in text_files],
        ("referensi", "txt", stopwords, None),
    ]
    files = [
        {
            "class": klass, "format": fmt, "path": p.relative_to(out_dir).as_posix(),
            "size_bytes": p.stat().st_size, "md5": _md5(p), "records": n,
        }
        for klass, fmt, p, n in outputs
    ]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "s3a://clean-data/novels_clean",
        "source_rows": n_clean,
        "source_rows_with_synopsis": n_text,
        "text_records": n_written_text,
        "files": files,
    }
    return manifest


def verify(manifest: dict) -> None:
    """
    Cocokkan ketiga berkas dengan sumbernya dengan MEMBACA ULANG dari disk.

    Ekspor yang menghitung baris saat menulis bisa saja melaporkan angka yang
    benar padahal berkasnya rusak; membaca ulang menangkap kasus itu.
    """
    out_dir = variety_dir()
    n_clean = manifest["source_rows"]

    with (out_dir / "structured" / "novel_metrics.csv").open(encoding="utf-8", newline="") as h:
        reader = csv.reader(h)
        header = next(reader)
        n_csv = sum(1 for _ in reader)
    if tuple(header) != STRUCTURED_COLUMNS:
        raise RuntimeError(f"Header CSV menyimpang dari kontrak: {header}")

    ids_json: set[int] = set()
    with (out_dir / "semistructured" / "novel_labels.jsonl").open(encoding="utf-8") as h:
        for line in h:
            ids_json.add(json.loads(line)["series_id"])

    ids_text: set[int] = set()
    n_text_lines = 0
    for shard in sorted((out_dir / "unstructured" / "synopsis").glob("synopsis-*.txt")):
        with shard.open(encoding="utf-8") as h:
            for line in h:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    raise RuntimeError(f"{shard.name}: baris bukan 3 kolom -> {line[:80]!r}")
                ids_text.add(int(parts[0]))
                n_text_lines += 1

    problems = []
    if n_csv != n_clean:
        problems.append(f"CSV {n_csv:,} != clean {n_clean:,}")
    if len(ids_json) != n_clean:
        problems.append(f"JSONL {len(ids_json):,} id unik != clean {n_clean:,}")
    if n_text_lines != manifest["text_records"] or len(ids_text) != n_text_lines:
        problems.append(f"TXT {n_text_lines:,} baris / {len(ids_text):,} id unik")
    if not ids_text <= ids_json:
        problems.append("ada id di TXT yang tidak ada di JSONL")
    if problems:
        raise RuntimeError("Verifikasi variety GAGAL: " + "; ".join(problems))

    logger.info(
        "Verifikasi baca-ulang LULUS | CSV=%s | JSONL=%s | TXT=%s baris (%d shard)",
        f"{n_csv:,}", f"{len(ids_json):,}", f"{n_text_lines:,}",
        sum(1 for f in manifest["files"] if f["path"].startswith("unstructured/synopsis")),
    )


def run(shards: int = DEFAULT_TEXT_SHARDS) -> dict:
    log_section(logger, "variety - pemisahan dataset per format")
    manifest = export(shards)
    for f in manifest["files"]:
        logger.info("%-18s %-6s %-40s %9.2f MB", f["class"], f["format"], f["path"],
                    f["size_bytes"] / 1_048_576)
    verify(manifest)
    (variety_dir() / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        logger.error("VARIETY GAGAL: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
