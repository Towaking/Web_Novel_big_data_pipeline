"""
INVENTARIS ELEMEN DATA — klasifikasi format & sensitivitas
Mengidentifikasi elemen setiap data dan formatnya: terstruktur,
semi-terstruktur, atau tidak terstruktur.

Modul ini **tidak** menebak klasifikasi dari tipe fisik kolom. Tipe fisik
menyesatkan: `genres` dan `release_frequency` sama-sama bertipe `string` di
CSV, tetapi yang pertama adalah daftar multi-nilai berpemisah koma dan yang
kedua adalah kalimat berpola (`"Every 0.9 Day(s)"`) yang harus diurai regex.
Keduanya semi-terstruktur karena alasan yang berbeda, dan tidak ada tipe fisik
yang bisa membedakannya dari `language` yang benar-benar skalar.

Karena itu klasifikasi ditulis eksplisit di `ELEMENT_REGISTRY` sebagai
**keputusan desain yang bisa diperdebatkan**, lengkap dengan alasannya, lalu
*statistiknya* diukur dari data sungguhan (kardinalitas, porsi sentinel,
panjang teks, contoh nilai). Yang dideklarasikan = penilaian; yang dihitung =
bukti.

Kriteria yang dipakai untuk membagi tiga kelas format:

  * **Terstruktur** — satu nilai atomik per sel, skema tetap, langsung bisa
    dipakai operator relasional (`=`, `<`, `SUM`) tanpa parsing apa pun.
  * **Semi-terstruktur** — nilainya punya struktur internal yang harus diurai
    lebih dulu: daftar berpemisah, pola angka+satuan, atau path hierarkis.
    Skemanya tetap, isinya tidak atomik.
  * **Tidak terstruktur** — prosa bahasa alami tanpa skema internal; baru
    berguna setelah melewati tokenisasi/vektorisasi (Fase 2 & 4).

Keluaran:
  * `s3a://curated-data/governance/data_elements` (Parquet) — dibaca dashboard
  * `Doc/generated/data_elements.md` — tabel Markdown inventaris dan rekap

Jalankan:
    docker compose exec pipeline python -m src.phases.data_catalog
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.common.config import get_settings
from src.common.logging_utils import get_logger, log_section
from src.common.spark_session import spark_session

logger = get_logger("data.catalog")

STRUCTURED = "Terstruktur"
SEMI = "Semi-terstruktur"
UNSTRUCTURED = "Tidak terstruktur"

#: Nilai yang menyamarkan "kosong" di sumber ini. Dataset tidak punya satu pun
#: NaN — kekosongan disandikan sebagai 0 atau "Unknown".
SENTINEL_NUMERIC = 0
SENTINEL_STRING = "Unknown"

#: Aturan sentinel per elemen, dideklarasikan per kolom: angka 0 berarti "tidak diketahui" pada `year` dan `chapter_count`,
#: tetapi nilai sah pada `is_completed` dan `reading_list_count`.
RULE_ZERO = "zero"  # 0 = kosong
RULE_UNKNOWN = "unknown"  # "Unknown" = kosong
RULE_EMPTY = "empty"  # string kosong = kosong
RULE_NONE = "none"  # kolom ini tidak punya sentinel


@dataclass(frozen=True)
class Element:
    """Satu elemen data beserta penilaian formatnya."""

    name: str
    source: str  # berkas asal
    physical_type: str  # tipe di CSV
    format_class: str  # STRUCTURED | SEMI | UNSTRUCTURED
    domain: str  # peran bisnis
    sensitivity: str  # untuk analisis privasi
    rationale: str  # kenapa diklasifikasikan begitu
    sentinel: str  # bagaimana kekosongan disandikan (prosa)
    sentinel_rule: str  # bagaimana kekosongan DIHITUNG (mesin)


MAIN = "novel_catalogue.csv"

#: Registry elemen. Urutannya mengikuti pengelompokan domain, bukan urutan
#: kolom di CSV, supaya tabel terbaca sebagai argumen, bukan dump.
ELEMENT_REGISTRY: tuple[Element, ...] = (
    Element(
        "series_id", MAIN, "integer", STRUCTURED, "Identitas",
        "Tidak ada", "Kunci primer alami: integer unik, atomik, tanpa struktur internal. "
        "Dipakai sebagai kunci dedup karena `title` tidak unik (ada judul kembar).",
        "tidak ada — selalu terisi", RULE_NONE,
    ),
    Element(
        "url", MAIN, "string", SEMI, "Identitas",
        "Publik", "Path hierarkis (`/series/<slug>/`) — nilainya membawa struktur "
        "yang bisa dipecah jadi segmen. Kunci penyatuan sumber scrape, bukan atribut analitik.",
        "tidak ada", RULE_EMPTY,
    ),
    Element(
        "title", MAIN, "string", UNSTRUCTURED, "Identitas",
        "Publik", "Teks bahasa alami bebas, tanpa skema internal. Pendek, tetapi tetap "
        "perlu normalisasi/tokenisasi sebelum dipakai model — ikut masuk korpus Fase 4.",
        "tidak ada", RULE_EMPTY,
    ),
    Element(
        "genres", MAIN, "string", SEMI, "Label",
        "Tidak ada", "Daftar multi-nilai berpemisah `, ` dalam satu sel. Skemanya tetap, "
        "isinya tidak atomik — harus di-split jadi array sebelum bisa di-`explode` "
        "atau dihitung frekuensinya. Menjadi label target Fase 4 Head A.",
        "string kosong bila tidak berlabel", RULE_EMPTY,
    ),
    Element(
        "tags", MAIN, "string", SEMI, "Label",
        "Tidak ada", "Sama seperti `genres` — daftar berpemisah koma, tetapi jauh lebih "
        "padat (median 12 tag). Label semantik ternormalisasi hasil kurasi komunitas.",
        f"{SENTINEL_STRING!r} pada baris tanpa tag", RULE_UNKNOWN,
    ),
    Element(
        "synopsis", MAIN, "string", UNSTRUCTURED, "Deskripsi",
        "Publik", "Sinopsis prosa bebas. Inti data tidak terstruktur di proyek ini: baru "
        "berguna setelah tokenisasi + TF-IDF/embedding di Fase 2 & 4.",
        "string kosong; <10 kata dianggap bukan deskripsi", RULE_EMPTY,
    ),
    Element(
        "type", MAIN, "string", STRUCTURED, "Dimensi",
        "Tidak ada", "Kategori tertutup berkardinalitas rendah, satu nilai per baris.",
        f"{SENTINEL_STRING!r} / 'N/A'", RULE_UNKNOWN,
    ),
    Element(
        "type_origin", MAIN, "string", STRUCTURED, "Dimensi",
        "Tidak ada", "Negara/asal terbitan — kategori atomik, dipakai sebagai sumbu agregasi Fase 3.",
        f"{SENTINEL_STRING!r} / 'N/A'", RULE_UNKNOWN,
    ),
    Element(
        "language", MAIN, "string", STRUCTURED, "Dimensi",
        "Tidak ada", "Kategori atomik berkardinalitas sangat rendah.",
        f"{SENTINEL_STRING!r} / 'N/A'", RULE_UNKNOWN,
    ),
    Element(
        "status_coo", MAIN, "string", SEMI, "Dimensi",
        "Tidak ada", "Campuran teks status dan pola angka+satuan (`\"182 Chapters (Completed)\"`). "
        "Fase 2 mengekstrak angka babnya dengan regex untuk memulihkan satu "
        "`chapter_count` yang terpotong pemisah ribuan — bukti nyata bahwa kolom ini "
        "menyimpan struktur di dalam string.",
        f"{SENTINEL_STRING!r} / 'N/A'", RULE_UNKNOWN,
    ),
    Element(
        "authors", MAIN, "string", SEMI, "Dimensi",
        "Nama orang (publik)", "Daftar multi-nilai berpemisah koma — satu novel bisa punya "
        "beberapa penulis. Berisi nama pena yang dipublikasikan platform, bukan identitas "
        "privat; tetap dicatat di sini karena relevan untuk Bab 3.3.",
        f"{SENTINEL_STRING!r} / 'N/A'", RULE_UNKNOWN,
    ),
    Element(
        "original_publisher", MAIN, "string", STRUCTURED, "Dimensi",
        "Nama entitas (publik)", "Nama badan usaha, satu nilai per baris. "
        "Penerbit yang tidak diketahui dijadikan null di Fase 2.",
        f"{SENTINEL_STRING!r} / 'N/A'", RULE_UNKNOWN,
    ),
    Element(
        "artists", MAIN, "string", STRUCTURED, "Dimensi",
        "Nama orang (publik)", "DIBUANG di Fase 2: hampir seluruhnya Unknown, tidak cukup "
        "sinyal untuk analitik apa pun.",
        f"{SENTINEL_STRING!r} / 'N/A'", RULE_UNKNOWN,
    ),
    Element(
        "english_publisher", MAIN, "string", STRUCTURED, "Dimensi",
        "Nama entitas (publik)", "DIBUANG di Fase 2: hampir seluruhnya Unknown.",
        f"{SENTINEL_STRING!r} / 'N/A'", RULE_UNKNOWN,
    ),
    Element(
        "year", MAIN, "integer", STRUCTURED, "Temporal",
        "Tidak ada", "Tahun terbit — skalar ordinal. Sentinel `0`, 'Unknown', dan 'N/A' TIDAK "
        "dapat dipulihkan dari kolom lain, jadi dijadikan null dan dikecualikan dari analisis "
        "temporal.",
        f"{SENTINEL_NUMERIC}, kosong, {SENTINEL_STRING!r}, atau 'N/A'", RULE_ZERO,
    ),
    Element(
        "release_frequency", MAIN, "string", SEMI, "Temporal",
        "Tidak ada", "Kalimat berpola `\"Every 0.9 Day(s)\"` — angka dan satuan tertanam "
        "di dalam teks. 100% baris dapat diurai regex menjadi `days_per_chapter`, "
        "dan hasilnya berkorelasi dengan rating sekuat jumlah bab. Contoh paling jelas "
        "kenapa 'bertipe string' tidak sama dengan 'tidak terstruktur'.",
        "tidak ada — selalu terisi, tetapi perlu parsing", RULE_EMPTY,
    ),
    Element(
        "chapter_count", MAIN, "integer", STRUCTURED, "Metrik",
        "Tidak ada", "Ukuran numerik aditif — langsung bisa di-`AVG`/`SUM`. "
        "Sentinel `0` dijadikan null lebih dulu, kalau tidak rata-rata "
        "panjang katalog turun palsu.",
        f"{SENTINEL_NUMERIC} atau kosong", RULE_ZERO,
    ),
    Element(
        "rating_score", MAIN, "double", STRUCTURED, "Metrik",
        "Tidak ada", "Skalar kontinu 0–5. Sentinel `0` berarti *belum pernah dinilai*, "
        "bukan *dinilai buruk* — dijadikan null, kalau tidak seluruh agregat rating bias ke bawah.",
        f"{SENTINEL_NUMERIC} atau kosong", RULE_ZERO,
    ),
    Element(
        "rating_votes", MAIN, "integer", STRUCTURED, "Metrik",
        "Tidak ada", "Jumlah suara — dipakai sebagai **bobot kepercayaan**, bukan metrik "
        "berdiri sendiri. Menjadi `v` pada rumus Bayesian weighted rating.",
        f"{SENTINEL_NUMERIC} sejalan dengan rating_score", RULE_ZERO,
    ),
    Element(
        "reading_list_count", MAIN, "integer", STRUCTURED, "Popularitas",
        "Agregat (bukan PII)", "Cacah agregat berapa pembaca memasukkan novel ke daftar baca. "
        "Ini angka gabungan — tidak ada baris per pengguna, sehingga tidak ada PII sama sekali "
        "(lihat catatan privasi di bawah).",
        f"{SENTINEL_NUMERIC} berarti nol pembaca, bukan hilang", RULE_NONE,
    ),
    Element(
        "activity_week_rank", MAIN, "integer", STRUCTURED, "Popularitas",
        "Agregat", "Peringkat aktivitas mingguan — ordinal, makin kecil makin populer.",
        f"{SENTINEL_NUMERIC} = tidak masuk peringkat", RULE_ZERO,
    ),
    Element(
        "activity_month_rank", MAIN, "integer", STRUCTURED, "Popularitas",
        "Agregat", "Peringkat aktivitas bulanan.",
        f"{SENTINEL_NUMERIC} = tidak masuk peringkat", RULE_ZERO,
    ),
    Element(
        "activity_all_rank", MAIN, "integer", STRUCTURED, "Popularitas",
        "Agregat", "Peringkat aktivitas sepanjang masa.",
        f"{SENTINEL_NUMERIC} = tidak masuk peringkat", RULE_ZERO,
    ),
    Element(
        "reading_list_month_rank", MAIN, "integer", STRUCTURED, "Popularitas",
        "Agregat", "Peringkat daftar baca bulanan.",
        f"{SENTINEL_NUMERIC} = tidak masuk peringkat", RULE_ZERO,
    ),
    Element(
        "reading_list_all_rank", MAIN, "integer", STRUCTURED, "Popularitas",
        "Agregat", "Peringkat daftar baca sepanjang masa.",
        f"{SENTINEL_NUMERIC} = tidak masuk peringkat", RULE_ZERO,
    ),
    Element(
        "is_completed", MAIN, "integer (0/1)", STRUCTURED, "Flag",
        "Tidak ada", "Boolean tersandi angka — atomik.",
        "tidak ada; 0 berarti belum tamat", RULE_NONE,
    ),
    Element(
        "licensed", MAIN, "integer (0/1)", STRUCTURED, "Flag",
        "Tidak ada", "Boolean tersandi angka. Relevan untuk Bab 3.2 (compliance): "
        "menandai judul yang punya lisensi resmi berbahasa Inggris.",
        "kosong bila halaman detail menulis N/A", RULE_NONE,
    ),
    Element(
        "completely_translated", MAIN, "integer (0/1)", STRUCTURED, "Flag",
        "Tidak ada", "Boolean tersandi angka.",
        "tidak ada", RULE_NONE,
    ),
)


def _sentinel_condition(name: str, rule: str):
    """
    Bangun kondisi "sel ini kosong" sesuai aturan yang dideklarasikan elemen.

    Null selalu dihitung kosong. Selebihnya bergantung aturan — dan itulah
    intinya: `is_completed = 0` bukan kekosongan, `year = 0` iya.
    """
    col = F.col(name)
    text_sentinel = (F.trim(col) == "") | F.lower(F.trim(col)).isin("unknown", "n/a")
    if rule == RULE_ZERO:
        # CSV dibaca tanpa inferSchema, jadi kolom numerik masih string; tanpa text_sentinel, 14% year bertulis
        # "Unknown"/"N/A" lolos (tercatat 1,93%, bukan 16,01%).
        return col.isNull() | text_sentinel | (col.cast("double") == SENTINEL_NUMERIC)
    if rule == RULE_UNKNOWN:
        return col.isNull() | text_sentinel
    if rule == RULE_EMPTY:
        return col.isNull() | (F.trim(col) == "")
    if rule == RULE_NONE:
        return col.isNull()
    raise ValueError(f"Aturan sentinel tidak dikenal untuk {name!r}: {rule!r}")


def _profile_frame(df: DataFrame, elements: list[Element]) -> dict[str, dict]:
    """
    Ukur semua kolom satu berkas: kardinalitas, porsi sentinel, panjang teks,
    dan satu contoh nilai.

    Dua job per berkas: satu agregasi statistik untuk semua kolom, satu untuk
    contoh nilai. Versi sebelumnya memakai satu agregasi + satu `limit(1)` PER
    KOLOM (~58 job Spark untuk 28 elemen); tiap job membayar overhead
    penjadwalan sendiri.

    Contoh nilai sengaja di job terpisah: di dalam agregasi yang memuat
    `countDistinct` ada shuffle, sehingga `first()` tidak lagi mengikuti urutan
    berkas dan contohnya berganti antar run.
    """
    exprs = [F.count(F.lit(1)).alias("n_rows")]
    sample_exprs = []
    for i, element in enumerate(elements):
        col = F.col(element.name)
        text = col.cast("string")
        sentinel_cond = _sentinel_condition(element.name, element.sentinel_rule)
        sample_ok = text.isNotNull() & (F.trim(text) != "") & ~F.lower(F.trim(text)).isin("unknown", "n/a")
        exprs += [
            F.countDistinct(col).alias(f"d{i}"),
            F.sum(sentinel_cond.cast("int")).alias(f"s{i}"),
            F.avg(F.length(text)).alias(f"a{i}"),
            F.max(F.length(text)).alias(f"m{i}"),
        ]
        # Berkas CSV multiLine = satu partisi dan agregasi tanpa shuffle, jadi
        # `first` mengikuti urutan berkas - sama dengan `limit(1)` sebelumnya.
        sample_exprs.append(F.first(F.when(sample_ok, text), ignorenulls=True).alias(f"v{i}"))
    row = df.agg(*exprs).first()
    samples = df.agg(*sample_exprs).first() if sample_exprs else {}

    n_rows = row["n_rows"] or 0
    profiles: dict[str, dict] = {}
    for i, element in enumerate(elements):
        n_sentinel = row[f"s{i}"] or 0
        sample = samples[f"v{i}"]
        if sample is None:
            sample = "-"
        else:
            sample = " ".join(sample.split())
            sample = sample[:70] + ("..." if len(sample) > 70 else "")
        profiles[element.name] = {
            "n_rows": n_rows,
            "n_distinct": row[f"d{i}"] or 0,
            "n_sentinel": n_sentinel,
            "pct_sentinel": round(100.0 * n_sentinel / n_rows, 2) if n_rows else 0.0,
            "avg_char_len": round(row[f"a{i}"] or 0.0, 1),
            "max_char_len": int(row[f"m{i}"] or 0),
            "sample_value": sample,
        }
    return profiles


def build_catalog(spark: SparkSession) -> DataFrame:
    """Gabungkan penilaian format (deklaratif) dengan statistik terukur."""
    settings = get_settings()
    minio = settings.minio

    main = spark.read.option("header", True).option("multiLine", True).option(
        "escape", '"'
    ).csv(minio.s3a_uri(minio.bucket_raw, settings.raw_object_key))

    present = [e for e in ELEMENT_REGISTRY if e.name in main.columns]
    for element in ELEMENT_REGISTRY:
        if element.name not in main.columns:
            logger.warning("Kolom %r tidak ada di %s — dilewati.", element.name, MAIN)
    profiles = _profile_frame(main, present)
    logger.info("Sumber dibaca: %s=%s baris", MAIN, f"{next(iter(profiles.values()), {}).get('n_rows', 0):,}")
    records: list[dict] = [{**asdict(e), **profiles[e.name]} for e in present]

    logger.info("Elemen terprofil: %d", len(records))

    # Peringatan bila ada kolom sumber yang belum masuk registry — supaya
    # penambahan kolom di kemudian hari tidak diam-diam lolos dari inventaris.
    declared = {e.name for e in ELEMENT_REGISTRY}
    undeclared = [c for c in main.columns if c not in declared]
    if undeclared:
        logger.warning("Kolom %s belum diklasifikasikan di ELEMENT_REGISTRY: %s", MAIN, ", ".join(undeclared))

    return spark.createDataFrame(records)


def summarise(catalog: DataFrame) -> DataFrame:
    """Rekap jumlah elemen per kelas format."""
    return (
        catalog.groupBy("format_class")
        .agg(
            F.count(F.lit(1)).alias("n_elements"),
            F.concat_ws(", ", F.array_sort(F.collect_list("name"))).alias("elements"),
        )
        .orderBy(F.col("n_elements").desc())
    )


def _markdown_table(rows: list[dict]) -> str:
    """Render inventaris jadi tabel Markdown siap tempel ke laporan."""
    header = (
        "| Elemen | Sumber | Tipe fisik | Kelas format | Domain | Unik | Sentinel | Contoh |\n"
        "|---|---|---|---|---|---:|---:|---|\n"
    )
    body = "".join(
        "| `{name}` | {source} | {physical_type} | **{format_class}** | {domain} "
        "| {n_distinct:,} | {pct_sentinel}% | {sample} |\n".format(
            sample=str(r["sample_value"]).replace("|", "\\|"), **r
        )
        for r in rows
    )
    return header + body


def write_markdown(rows: list[dict], summary_rows: list[dict]) -> None:
    """
    Tulis `Doc/generated/data_elements.md`.

    Berkas ini **dihasilkan mesin** dan berisi fakta terukur saja — tabel
    inventaris dan rekap. Interpretasi, argumentasi, dan pemilihan model data
    tetap ditulis manusia di laporan; jangan menambahkan prosa analitis ke sini
    karena akan tertimpa setiap kali modul dijalankan ulang.
    """
    settings = get_settings()
    out_dir = settings.paths.doc_dir / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "data_elements.md"

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Inventaris Elemen Data (dihasilkan otomatis)",
        "",
        f"> Dihasilkan `src/phases/data_catalog.py` pada {stamp}. **Jangan diedit "
        "manual** — jalankan ulang modulnya. Angka di sini terukur dari data; "
        "klasifikasi formatnya adalah keputusan desain yang alasannya ada di "
        "`ELEMENT_REGISTRY`.",
        "",
        "## Rekap per kelas format",
        "",
        "| Kelas format | Jumlah elemen |",
        "|---|---:|",
    ]
    lines += [f"| {r['format_class']} | {r['n_elements']} |" for r in summary_rows]
    lines += ["", "## Inventaris lengkap", "", _markdown_table(rows), "## Alasan klasifikasi", ""]
    for r in rows:
        lines.append(f"- **`{r['name']}`** → {r['format_class']}. {r['rationale']}")
    lines += [
        "",
        "## Sentinel value per elemen",
        "",
        "| Elemen | Penyandian kekosongan |",
        "|---|---|",
    ]
    lines += [f"| `{r['name']}` | {r['sentinel']} |" for r in rows]
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Inventaris Markdown ditulis -> %s", out_path)


def run() -> dict:
    """Bangun inventaris elemen, simpan ke curated zone + Doc/generated/."""
    log_section(logger, "inventaris elemen data & klasifikasi format")
    settings = get_settings()

    with spark_session("novel-data-catalog") as spark:
        catalog = build_catalog(spark).cache()
        summary = summarise(catalog)

        catalog.select(
            "name", "source", "format_class", "domain", "n_distinct", "pct_sentinel"
        ).show(40, truncate=28)
        summary.select("format_class", "n_elements").show(truncate=False)

        uri = settings.minio.s3a_uri(settings.minio.bucket_curated, "governance/data_elements")
        catalog.coalesce(1).write.mode("overwrite").parquet(uri)
        logger.info("Inventaris tersimpan -> %s", uri)

        summary_uri = settings.minio.s3a_uri(
            settings.minio.bucket_curated, "governance/data_element_summary"
        )
        summary.coalesce(1).write.mode("overwrite").parquet(summary_uri)

        rows = [r.asDict() for r in catalog.collect()]
        summary_rows = [r.asDict() for r in summary.collect()]
        write_markdown(rows, summary_rows)

        counts = {r["format_class"]: r["n_elements"] for r in summary_rows}

    logger.info("INVENTARIS ELEMEN SELESAI — %s", counts)
    return {"n_elements": len(rows), "by_format": counts}


def main() -> int:
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        logger.error("Inventaris elemen gagal: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
