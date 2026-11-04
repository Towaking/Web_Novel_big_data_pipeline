"""
MODEL DATA — skema dimensional + proyeksi graf
Memilih model data dan operasi yang sesuai dengan karakteristik masing-masing data.

Argumen modul ini: **satu model data saja tidak cukup untuk katalog ini**,
karena ketiga kelas format (lihat `src/phases/data_catalog.py`) menuntut
operasi yang berbeda.

| Karakteristik data | Model yang dipilih | Operasi yang jadi murah |
|---|---|---|
| Metrik & dimensi atomik (terstruktur) | **Star schema** — `fact_novel` + `dim_*` | filter, join berbintang, `GROUP BY`, roll-up OLAP |
| Multi-nilai `genres`/`tags`/`authors` (semi-terstruktur) | **Bridge table** hasil `explode` | `GROUP BY` per nilai tanpa `LIKE '%...%'` |
| Relasi antar-entitas (novel–genre–tag–penulis) | **Property graph** (`graph_nodes` + `graph_edges`) | traversal multi-hop, ko-okurensi, kedekatan |
| Sinopsis prosa (tidak terstruktur) | **Matriks fitur** (TF-IDF/embedding, Fase 4) | kemiripan kosinus, klasifikasi |

Kenapa bridge table, bukan menyimpan array saja: array Spark memang bisa
di-`explode` saat kueri, tetapi materialisasinya sekali di sini membuat setiap
agregasi genre di Fase 3/5 berhenti membayar ongkos `explode` berulang — dan,
lebih penting, memberi kolom yang bisa dijadikan kunci join dari model graf.

Kenapa graf tidak dijalankan di database graf terpisah: pertanyaan graf pada
proyek ini (ko-okurensi genre, jangkauan penulis) semuanya berkedalaman <= 2
hop pada puluhan ribu simpul novel. Self-join pada tabel sisi menyelesaikannya dalam
hitungan detik tanpa menambah satu service pun. Model grafnya tetap
dimaterialisasi dalam bentuk `nodes`/`edges` standar supaya, kalau kebutuhan
berkembang ke traversal dalam (jalur terpendek, komunitas, PageRank), isinya
tinggal dimuat ke Neo4j/GraphFrames tanpa memodelkan ulang.

Keluaran (semua Parquet di `s3a://curated-data/model/`):
  dim_novel, dim_genre, dim_tag, dim_author, dim_publisher, dim_time,
  bridge_novel_genre, bridge_novel_tag, bridge_novel_author,
  fact_novel, graph_nodes, graph_edges,
  op_olap_genre_decade, op_graph_genre_cooccurrence, op_graph_author_reach

Jalankan:
    docker compose exec pipeline python -m src.phases.data_model
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.common.config import get_settings
from src.common.logging_utils import get_logger, log_section
from src.common.spark_session import spark_session

logger = get_logger("data.model")

#: Sentinel string yang tidak boleh masuk tabel dimensi sebagai entitas nyata.
UNKNOWN = "Unknown"

#: Prefiks id simpul graf. Node id harus unik LINTAS tipe — tanpa prefiks,
#: genre_key 7 dan tag_key 7 akan bertabrakan jadi satu simpul.
NODE_PREFIX = {"Novel": "N", "Genre": "G", "Tag": "T", "Author": "A"}

#: Kolom novels_clean yang tidak dipakai model data (teks & token Fase 2).
HEAVY_TEXT_COLUMNS = ("synopsis", "corpus_raw", "corpus_clean", "tokens", "synopsis_tokens", "synopsis_clean")

#: Rentang aksara CJK + Hangul + Kana. Dipakai untuk membedakan nama asli dari
#: romanisasinya saat resolusi entitas penulis (lihat `resolve_author_aliases`).
_CJK = re.compile(r"[㐀-鿿가-힯぀-ヿ]")

#: Label hasil resolusi, ikut disimpan di `dim_author` supaya keputusannya
#: bisa diaudit — bukan disembunyikan di balik angka akhir.
RES_SINGLE = "tunggal"  # nama tidak punya pasangan alias
RES_MERGED = "digabung"  # komponen 1 nama asli + n romanisasi
RES_AMBIGUOUS = "ambigu"  # >=2 nama beraksara asli: sengaja TIDAK digabung


def _surrogate(df: DataFrame, natural_key: str, key_name: str) -> DataFrame:
    """
    Beri kunci pengganti (surrogate key) yang deterministik.

    Diurutkan `n_novels DESC, <natural_key> ASC` — bukan
    `monotonically_increasing_id()` — supaya kunci yang sama keluar di setiap
    eksekusi. Kunci yang bergoyang antar-run membuat artefak graf tidak bisa
    dibandingkan lintas waktu dan menyulitkan debugging.
    """
    window = Window.orderBy(F.col("n_novels").desc(), F.col(natural_key).asc())
    return df.withColumn(key_name, F.row_number().over(window))


def _explode_dimension(
    novels: DataFrame, array_col: str, value_name: str, key_name: str
) -> tuple[DataFrame, DataFrame]:
    """
    Ubah satu kolom multi-nilai menjadi pasangan (tabel dimensi, bridge table).

    Inilah operasi inti untuk data semi-terstruktur: nilai yang tadinya
    terkubur di dalam satu sel dinaikkan menjadi baris tersendiri, sehingga
    bisa dihitung, di-join, dan dijadikan simpul graf.
    """
    pairs = (
        novels.select("series_id", F.explode(F.col(array_col)).alias(value_name))
        .filter(F.col(value_name).isNotNull() & (F.trim(F.col(value_name)) != ""))
        .filter(F.col(value_name) != UNKNOWN)
        .distinct()
    )

    dim = _surrogate(
        pairs.groupBy(value_name).agg(F.count(F.lit(1)).alias("n_novels")),
        value_name,
        key_name,
    ).select(key_name, value_name, "n_novels")

    bridge = pairs.join(dim.select(key_name, value_name), on=value_name, how="inner").select(
        "series_id", key_name
    )
    return dim, bridge


def resolve_author_aliases(spark: SparkSession, author_pairs: DataFrame) -> DataFrame:
    """
    Satukan nama penulis yang sebenarnya satu orang dalam dua aksara.

    **Masalahnya.** Kolom `authors` bukan selalu daftar beberapa penulis. Pada
    1.494 novel (34,6%) isinya satu orang yang ditulis dua kali — nama asli dan
    romanisasinya: `"Mizume, ミズメ"`, `"Chae Yeon-sil, 채연실"`. Diperlakukan
    apa adanya, `dim_author` menggelembung 5.439 entitas untuk 4.461 novel dan
    **setiap** analitik tingkat penulis menghitung ganda.

    **Aturannya.** Dua nama dianggap alias bila (a) salah satu beraksara
    CJK/Hangul/Kana dan yang lain tidak, DAN (b) yang jarang tidak pernah muncul
    tanpa yang sering (`co_occurrence == min(count_a, count_b)`). Syarat (b)
    memakai *containment*, bukan "selalu bersama", karena nama asli sering
    muncul sendirian di novel lain yang romanisasinya tidak dicatat.

    **Batas yang sengaja dipasang.** Menutup alias secara transitif akan salah:
    trio penulis yang selalu berkarya bersama (`"Aono Hakuto, Imai Santarou,
    Maraiya Mū, マライヤ・ムー, 今井三太郎, 蒼乃白兎"`) akan tertarik jadi satu
    orang. Karena itu komponen hanya digabung bila mengandung **tepat satu**
    nama beraksara asli. Komponen dengan >=2 nama asli (36 komponen, 141 nama)
    dibiarkan terpisah dan ditandai `ambigu` — termasuk kasus yang sebenarnya
    sama seperti `荆柯守`/`荊柯守` (sederhana vs tradisional). Melewatkan
    penggabungan yang benar lebih murah daripada melebur dua orang berbeda.

    Grafnya diselesaikan di driver, bukan lewat job Spark iteratif: simpulnya
    5.439 dan sisinya 1.771 — muat di memori driver berkali lipat, dan
    connected-components terdistribusi di skala ini lebih mahal daripada
    hasilnya.
    """
    counts = Counter()
    for row in author_pairs.groupBy("author").agg(F.count(F.lit(1)).alias("n")).collect():
        counts[row["author"]] = row["n"]

    # Ko-okurensi dihitung di Spark (join pada 6k baris), hasilnya kecil.
    left = author_pairs.select("series_id", F.col("author").alias("a"))
    right = author_pairs.select("series_id", F.col("author").alias("b"))
    cooc = (
        left.join(right, on="series_id")
        .filter(F.col("a") < F.col("b"))
        .groupBy("a", "b")
        .agg(F.count(F.lit(1)).alias("n_shared"))
        .collect()
    )

    parent: dict[str, str] = {name: name for name in counts}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_edges = 0
    for row in cooc:
        a, b, shared = row["a"], row["b"], row["n_shared"]
        different_script = bool(_CJK.search(a)) != bool(_CJK.search(b))
        contained = shared == min(counts[a], counts[b])
        if not (different_script and contained):
            continue
        n_edges += 1
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    components: dict[str, list[str]] = defaultdict(list)
    for name in counts:
        components[find(name)].append(name)

    records: list[dict] = []
    stats = Counter()
    for members in components.values():
        native = [m for m in members if _CJK.search(m)]
        if len(members) == 1:
            resolution = RES_SINGLE
        elif len(native) == 1:
            resolution = RES_MERGED
        else:
            resolution = RES_AMBIGUOUS

        if resolution == RES_MERGED:
            # Kanonik = romanisasi paling sering dipakai (paling terbaca di
            # dashboard/laporan); nama asli & romanisasi lain jadi alias.
            romanised = [m for m in members if not _CJK.search(m)]
            canonical = sorted(romanised, key=lambda m: (-counts[m], m))[0]
            aliases = sorted(m for m in members if m != canonical)
            for member in members:
                records.append(
                    {
                        "author_name": member,
                        "canonical": canonical,
                        "resolution": resolution,
                        "aliases": ", ".join(aliases),
                    }
                )
            stats[resolution] += 1
        else:
            # Tunggal & ambigu: setiap nama tetap jadi entitasnya sendiri.
            for member in members:
                records.append(
                    {
                        "author_name": member,
                        "canonical": member,
                        "resolution": resolution,
                        "aliases": "",
                    }
                )
            stats[resolution] += 1

    n_out = len({r["canonical"] for r in records})
    logger.info(
        "Resolusi penulis: %s nama -> %s entitas | sisi alias=%s | "
        "komponen: tunggal=%s digabung=%s ambigu=%s",
        f"{len(counts):,}", f"{n_out:,}", f"{n_edges:,}",
        stats[RES_SINGLE], stats[RES_MERGED], stats[RES_AMBIGUOUS],
    )
    return spark.createDataFrame(records)


def build_author_dimension(
    spark: SparkSession, novels: DataFrame
) -> tuple[DataFrame, DataFrame]:
    """Tabel dimensi penulis (sudah diresolusi) + bridge-nya."""
    # `authors` masih string berpemisah koma di clean zone (Fase 2 hanya
    # memecah genres & tags). Dipecah di sini dengan aturan yang sama.
    pairs = (
        novels.select(
            "series_id",
            F.explode(
                F.array_distinct(
                    F.filter(
                        F.transform(F.split(F.col("authors"), ","), lambda x: F.trim(x)),
                        lambda x: F.length(x) > 0,
                    )
                )
            ).alias("author"),
        )
        .filter(F.col("author") != UNKNOWN)
        .distinct()
        .cache()
    )

    mapping = resolve_author_aliases(spark, pairs)

    resolved = (
        pairs.join(mapping, pairs["author"] == mapping["author_name"], "inner")
        .select("series_id", F.col("canonical").alias("author"), "resolution", "aliases")
        .distinct()
    )

    dim = _surrogate(
        resolved.groupBy("author", "resolution", "aliases").agg(
            F.count(F.lit(1)).alias("n_novels")
        ),
        "author",
        "author_key",
    ).select(
        "author_key",
        "author",
        "n_novels",
        "resolution",
        "aliases",
        F.when(F.col("aliases") == "", F.lit(0))
        .otherwise(F.size(F.split(F.col("aliases"), ", ")))
        .alias("n_aliases"),
    )

    bridge = resolved.join(dim.select("author_key", "author"), on="author", how="inner").select(
        "series_id", "author_key"
    )
    return dim, bridge


def build_dimensions(spark: SparkSession, novels: DataFrame) -> dict[str, DataFrame]:
    """Bangun seluruh tabel dimensi + bridge dari data bersih."""
    dim_genre, bridge_genre = _explode_dimension(novels, "genre_list", "genre", "genre_key")
    dim_tag, bridge_tag = _explode_dimension(novels, "tag_list", "tag", "tag_key")
    # Penulis tidak bisa lewat `_explode_dimension` biasa: nilainya perlu
    # resolusi entitas lebih dulu (satu orang, dua aksara).
    dim_author, bridge_author = build_author_dimension(spark, novels)

    dim_publisher = _surrogate(
        novels.filter(F.col("original_publisher").isNotNull())
        .groupBy(F.col("original_publisher").alias("publisher"))
        .agg(F.count(F.lit(1)).alias("n_novels")),
        "publisher",
        "publisher_key",
    ).select("publisher_key", "publisher", "n_novels")

    # dim_time sengaja dibatasi pada tahun yang valid: `year = 0` adalah
    # sentinel (22,66% baris) dan tidak boleh jadi anggota dimensi waktu.
    dim_time = (
        novels.filter(F.col("year").isNotNull())
        .groupBy("year", "decade")
        .agg(F.count(F.lit(1)).alias("n_novels"))
        .orderBy("year")
    )

    dim_novel = novels.select(
        "series_id",
        "title",
        "url",
        "type",
        "type_origin",
        "language",
        "status_coo",
        "is_completed",
        "licensed",
        "completely_translated",
        "primary_genre",
        "genre_combo",
        "length_bucket",
        "release_pace",
        "has_synopsis",
        "has_text_signal",
        "n_genres",
        "n_tags",
    )

    return {
        "dim_novel": dim_novel,
        "dim_genre": dim_genre,
        "dim_tag": dim_tag,
        "dim_author": dim_author,
        "dim_publisher": dim_publisher,
        "dim_time": dim_time,
        "bridge_novel_genre": bridge_genre,
        "bridge_novel_tag": bridge_tag,
        "bridge_novel_author": bridge_author,
    }


def build_fact(novels: DataFrame, dim_publisher: DataFrame) -> DataFrame:
    """
    Tabel fakta, grain = satu baris per novel (`series_id`).

    Grain ditulis eksplisit karena inilah keputusan terpenting sebuah star
    schema. Grain per-novel dipilih (bukan per novel-genre) supaya ukuran
    seperti `rating_votes` tidak tergandakan saat novel bergenre banyak —
    kesalahan klasik yang membuat total suara katalog membengkak 2-3x.
    Analisis per genre tetap bisa dilakukan lewat `bridge_novel_genre`,
    dengan konsekuensi yang disadari: hasilnya *additive across novels*,
    bukan across genres.
    """
    return (
        novels.join(
            dim_publisher.select("publisher_key", F.col("publisher").alias("original_publisher")),
            on="original_publisher",
            how="left",
        )
        .select(
            "series_id",
            "publisher_key",
            "year",
            "decade",
            # ukuran aditif
            "chapter_count",
            "rating_votes",
            "reading_list_count",
            # ukuran non-aditif (rata-rata; jangan di-SUM)
            "rating_score",
            "weighted_rating",
            "days_per_chapter",
            "popularity_score",
            # ukuran teks
            "synopsis_word_count",
            "token_count",
            # penanda populasi
            "is_rated_reliable",
            "has_synopsis",
        )
    )


def build_graph(dims: dict[str, DataFrame]) -> tuple[DataFrame, DataFrame]:
    """
    Proyeksikan model dimensional menjadi property graph (nodes + edges).

    Bentuknya sengaja mengikuti konvensi umum database graf — satu tabel simpul
    berlabel dan satu tabel sisi bertipe relasi — supaya bisa dimuat langsung
    ke Neo4j (`LOAD CSV`) atau GraphFrames tanpa transformasi tambahan.
    """
    def node(df: DataFrame, key: str, label_col: str, node_type: str) -> DataFrame:
        prefix = NODE_PREFIX[node_type]
        return df.select(
            F.concat(F.lit(prefix + ":"), F.col(key).cast("string")).alias("node_id"),
            F.lit(node_type).alias("node_type"),
            F.col(label_col).cast("string").alias("label"),
            F.col("n_novels").cast("long").alias("weight"),
        )

    nodes = (
        node(
            dims["dim_novel"].withColumn("n_novels", F.lit(1)),
            "series_id", "title", "Novel",
        )
        .unionByName(node(dims["dim_genre"], "genre_key", "genre", "Genre"))
        .unionByName(node(dims["dim_tag"], "tag_key", "tag", "Tag"))
        .unionByName(node(dims["dim_author"], "author_key", "author", "Author"))
    )

    def edge(bridge: DataFrame, key: str, node_type: str, rel: str) -> DataFrame:
        prefix = NODE_PREFIX[node_type]
        return bridge.select(
            F.concat(F.lit("N:"), F.col("series_id").cast("string")).alias("src_id"),
            F.concat(F.lit(prefix + ":"), F.col(key).cast("string")).alias("dst_id"),
            F.lit(rel).alias("rel_type"),
        )

    edges = (
        edge(dims["bridge_novel_genre"], "genre_key", "Genre", "HAS_GENRE")
        .unionByName(edge(dims["bridge_novel_tag"], "tag_key", "Tag", "HAS_TAG"))
        .unionByName(edge(dims["bridge_novel_author"], "author_key", "Author", "WRITTEN_BY"))
    )
    return nodes, edges


def op_olap_genre_decade(spark: SparkSession) -> DataFrame:
    """
    Operasi relasional/OLAP di atas star schema: roll-up genre x dekade.

    Ini pertanyaan yang star schema jawab dengan baik — join berbintang lalu
    agregasi. Perhatikan `AVG(weighted_rating)`, bukan `SUM`: rating adalah
    ukuran non-aditif, menjumlahkannya lintas novel tidak punya arti.
    """
    return spark.sql(
        """
        SELECT  g.genre,
                f.decade,
                COUNT(*)                                   AS n_novels,
                ROUND(AVG(f.weighted_rating), 4)           AS avg_weighted_rating,
                ROUND(AVG(f.chapter_count), 1)             AS avg_chapters,
                SUM(f.rating_votes)                        AS total_votes
        FROM        fact_novel        f
        JOIN        bridge_novel_genre b ON b.series_id = f.series_id
        JOIN        dim_genre          g ON g.genre_key  = b.genre_key
        WHERE   f.decade IS NOT NULL
        GROUP BY    g.genre, f.decade
        HAVING      COUNT(*) >= 10
        ORDER BY    f.decade DESC, avg_weighted_rating DESC
        """
    )


def op_graph_genre_cooccurrence(spark: SparkSession) -> DataFrame:
    """
    Operasi graf: traversal 2-hop `Genre <- Novel -> Genre`.

    Pertanyaannya — "genre apa yang cenderung muncul bersama, dan seberapa
    kuat?" — tidak punya bentuk alami di star schema; ia butuh menelusuri sisi,
    bukan mengagregasi ukuran. Kekuatannya diukur dengan **koefisien Jaccard**
    (irisan / gabungan), bukan cacah mentah, supaya genre besar seperti
    `Fantasy` tidak otomatis tampak dekat dengan segalanya.
    """
    return spark.sql(
        """
        WITH pair AS (
            SELECT  a.genre_key AS src_key,
                    b.genre_key AS dst_key,
                    COUNT(*)    AS n_shared
            FROM        bridge_novel_genre a
            JOIN        bridge_novel_genre b
                     ON a.series_id = b.series_id
                    AND a.genre_key < b.genre_key   -- hindari duplikat & self-loop
            GROUP BY    a.genre_key, b.genre_key
        )
        SELECT  gs.genre                                    AS genre_a,
                gd.genre                                    AS genre_b,
                p.n_shared,
                gs.n_novels                                 AS n_a,
                gd.n_novels                                 AS n_b,
                ROUND(p.n_shared / (gs.n_novels + gd.n_novels - p.n_shared), 4)
                                                            AS jaccard
        FROM        pair      p
        JOIN        dim_genre gs ON gs.genre_key = p.src_key
        JOIN        dim_genre gd ON gd.genre_key = p.dst_key
        WHERE       p.n_shared >= 20
        ORDER BY    jaccard DESC
        """
    )


def op_graph_author_reach(spark: SparkSession) -> DataFrame:
    """
    Operasi graf: traversal 2-hop `Author -> Novel -> Genre`.

    Menjawab "seberapa lebar jangkauan genre seorang penulis" — sekali lagi
    pertanyaan tentang bentuk relasi, bukan tentang ukuran numerik.
    """
    # Rating dan rentang genre dihitung pada grain masing-masing: satu join Author -> Novel -> Genre -> Fakta
    # memberi AVG(weighted_rating) bobot jumlah genre tiap novel.
    return spark.sql(
        """
        WITH rating AS (
            SELECT      ba.author_key, AVG(f.weighted_rating) AS avg_wr
            FROM        bridge_novel_author ba
            JOIN        fact_novel          f  ON f.series_id = ba.series_id
            GROUP BY    ba.author_key
        ),
        span AS (
            SELECT      ba.author_key,
                        COUNT(DISTINCT g.genre)                           AS n_distinct_genres,
                        CONCAT_WS(', ', ARRAY_SORT(COLLECT_SET(g.genre))) AS genre_span
            FROM        bridge_novel_author ba
            JOIN        bridge_novel_genre  bg ON bg.series_id = ba.series_id
            JOIN        dim_genre           g  ON g.genre_key  = bg.genre_key
            GROUP BY    ba.author_key
        )
        SELECT  a.author,
                a.n_novels,
                s.n_distinct_genres,
                ROUND(r.avg_wr, 4)  AS avg_weighted_rating,
                s.genre_span
        FROM        dim_author a
        JOIN        span       s ON s.author_key = a.author_key
        LEFT JOIN   rating     r ON r.author_key = a.author_key
        WHERE       a.n_novels >= 3
        ORDER BY    s.n_distinct_genres DESC, a.n_novels DESC
        """
    )


def save(df: DataFrame, name: str, preview: int = 0, keep: bool = True) -> int:
    """
    Materialisasi satu tabel model ke curated zone; kembalikan cacah barisnya.

    `keep=False` untuk tabel yang tidak dirujuk lagi sesudah ditulis (proyeksi
    graf, operasi contoh) - cache-nya dilepas, bukan menumpuk sampai sesi
    berakhir. Dimensi/bridge tetap di-cache karena dipakai ulang oleh view SQL.
    """
    settings = get_settings()
    uri = settings.minio.s3a_uri(settings.minio.bucket_curated, f"model/{name}")
    df = df.cache()
    rows = df.count()
    df.coalesce(1).write.mode("overwrite").parquet(uri)
    logger.info("[%-26s] %7s baris -> %s", name, f"{rows:,}", uri)
    if preview:
        df.show(preview, truncate=40)
    if not keep:
        df.unpersist()
    return rows


def run() -> dict:
    """Bangun star schema + proyeksi graf, lalu jalankan operasi contohnya."""
    log_section(logger, "model data — star schema & proyeksi graf")
    settings = get_settings()
    counts: dict[str, int] = {}

    with spark_session("novel-data-model") as spark:
        clean_uri = settings.minio.s3a_uri(settings.minio.bucket_clean, "novels_clean")
        logger.info("Membaca data bersih dari %s", clean_uri)
        # Kolom teks berat (sinopsis + empat salinan korpus) tidak dipakai model
        # data; tanpa drop, semuanya ikut tertampung di cache.
        novels = spark.read.parquet(clean_uri).drop(*HEAVY_TEXT_COLUMNS).cache()
        logger.info("Sumber: %s baris", f"{novels.count():,}")

        dims = build_dimensions(spark, novels)
        fact = build_fact(novels, dims["dim_publisher"])
        nodes, edges = build_graph(dims)

        logger.info(">> Tabel dimensi & bridge")
        for name in (
            "dim_novel", "dim_genre", "dim_tag", "dim_author",
            "dim_publisher", "dim_time",
            "bridge_novel_genre", "bridge_novel_tag", "bridge_novel_author",
        ):
            counts[name] = save(dims[name], name)

        logger.info(">> Tabel fakta")
        counts["fact_novel"] = save(fact, "fact_novel")

        logger.info(">> Proyeksi graf")
        counts["graph_nodes"] = save(nodes, "graph_nodes", keep=False)
        counts["graph_edges"] = save(edges, "graph_edges", keep=False)

        # Daftarkan sebagai view supaya operasinya ditulis SQL mentah,
        #     konsisten dengan gaya Fase 3 dan mudah diaudit.
        for name, frame in {**dims, "fact_novel": fact}.items():
            frame.createOrReplaceTempView(name)

        logger.info(">> Operasi contoh")
        olap = op_olap_genre_decade(spark)
        counts["op_olap_genre_decade"] = save(olap, "op_olap_genre_decade", preview=8, keep=False)

        cooc = op_graph_genre_cooccurrence(spark)
        counts["op_graph_genre_cooccurrence"] = save(
            cooc, "op_graph_genre_cooccurrence", preview=8, keep=False
        )

        reach = op_graph_author_reach(spark)
        counts["op_graph_author_reach"] = save(reach, "op_graph_author_reach", preview=8, keep=False)

    logger.info("MODEL DATA SELESAI — %d tabel", len(counts))
    return counts


def main() -> int:
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        logger.error("Pembangunan model data gagal: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
