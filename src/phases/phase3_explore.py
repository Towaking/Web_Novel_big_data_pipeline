"""
FASE 3 — EXPLORE (Analytics & Querying)
Membaca Parquet bersih dari `s3a://clean-data/`, mendaftarkannya sebagai
temporary view, lalu menjalankan **kueri Spark SQL mentah** untuk lima poin
analitik yang diminta. Setiap hasil disimpan kembali ke
`s3a://curated-data/analytics/<nama>` sebagai Parquet.

Kenapa hasilnya di-materialisasi, bukan dihitung ulang saat dashboard dibuka:
dashboard Streamlit harus responsif (milidetik), sedangkan menyalakan job
Spark per interaksi memakan detik. Fase 3 = *compute once*, Fase 5 = *read many*.

Dua aturan analitik yang berlaku di semua kueri:
  * Rating dipakai dalam bentuk `weighted_rating` (Bayesian), dan populasi
    peringkat dibatasi `is_rated_reliable` (>= 10 suara).
  * Setiap kueri melaporkan `n` yang dipakai supaya pembaca tahu berapa baris
    yang dikecualikan — angka rata-rata tanpa konteks populasi menyesatkan.

Jalankan:
    docker compose exec pipeline python -m src.phases.phase3_explore
"""

from __future__ import annotations

import sys

from pyspark.sql import DataFrame, SparkSession

from src.common.config import get_settings
from src.common.logging_utils import get_logger, log_section
from src.common.spark_session import spark_session

logger = get_logger("phase3.explore")

#: Minimum jumlah novel per kelompok agar layak masuk peringkat.
#: Tanpa ambang ini, genre berisi 2 novel bisa menempati puncak peringkat rating.
MIN_GROUP_SUPPORT = 25

#: Kolom novels_clean yang dipakai Fase 3, diproyeksikan sebelum cache agar sinopsis mentah dan korpus teks tidak ikut.
USED_COLUMNS = (
    "series_id", "title", "url", "primary_genre", "genre_list", "genre_combo", "n_genres",
    "tag_list", "n_tags", "type", "type_origin", "language", "year", "decade", "status_coo",
    "chapter_count", "length_bucket", "is_completed", "rating_score", "rating_votes",
    "weighted_rating", "is_rated_reliable", "reading_list_count", "popularity_score",
    "token_count", "synopsis_tokens", "synopsis_word_count", "has_synopsis",
    "days_per_chapter", "release_pace",
)


def register_views(spark: SparkSession) -> DataFrame:
    """Daftarkan view utama + view hasil explode untuk analisis multi-nilai."""
    settings = get_settings()
    clean_uri = settings.minio.s3a_uri(settings.minio.bucket_clean, "novels_clean")
    logger.info("Membaca data bersih dari %s", clean_uri)

    novels = spark.read.parquet(clean_uri).select(*USED_COLUMNS).cache()
    novels.createOrReplaceTempView("novels")

    # View turunan dibuat lewat SQL agar seluruh layer analitik konsisten berbahasa SQL.
    spark.sql(
        """
        CREATE OR REPLACE TEMP VIEW novel_genres AS
        SELECT series_id, title, weighted_rating, rating_votes, chapter_count,
               type_origin, is_rated_reliable, explode(genre_list) AS genre
        FROM novels
        WHERE size(genre_list) > 0
        """
    )
    spark.sql(
        """
        CREATE OR REPLACE TEMP VIEW novel_tags AS
        SELECT series_id, weighted_rating, is_rated_reliable, explode(tag_list) AS tag
        FROM novels
        WHERE size(tag_list) > 0
        """
    )
    # Token sinopsis dipisah dari korpus gabungan: analitik prosa harus
    # mengukur kosakata cerita, bukan frasa katalog yang berasal dari tag.
    spark.sql(
        """
        CREATE OR REPLACE TEMP VIEW novel_synopsis_tokens AS
        SELECT series_id, explode(synopsis_tokens) AS token
        FROM novels
        WHERE has_synopsis
        """
    )

    logger.info("View terdaftar: novels (%s baris), novel_genres, novel_tags, novel_synopsis_tokens",
                f"{novels.count():,}")
    return novels


def save(df: DataFrame, name: str, preview: int = 8) -> None:
    """Materialisasi hasil kueri ke curated zone + tampilkan cuplikan di log."""
    settings = get_settings()
    uri = settings.minio.s3a_uri(settings.minio.bucket_curated, f"analytics/{name}")
    df.coalesce(1).write.mode("overwrite").parquet(uri)
    logger.info("[%s] tersimpan -> %s", name, uri)
    if preview:
        # Cuplikan dibaca dari berkas yang baru ditulis (satu file kecil), bukan
        # `df.show()` yang menjalankan ulang seluruh kueri (self-join, percentile).
        df.sparkSession.read.parquet(uri).show(preview, truncate=45)


def analytic_1_descriptive(spark: SparkSession) -> None:
    logger.info(">> Analitik 1: statistik deskriptif & outlier")

    # Statistik ringkas untuk kedua metrik numerik inti. UNION ALL dipakai agar
    # keduanya keluar dalam satu tabel bertipe sama (rapi untuk dashboard).
    stats = spark.sql(
        """
        SELECT 'rating_score'   AS metric,
               COUNT(rating_score)                             AS n,
               ROUND(AVG(rating_score), 4)                     AS mean,
               ROUND(PERCENTILE_APPROX(rating_score, 0.5), 4)  AS median,
               ROUND(STDDEV(rating_score), 4)                  AS stddev,
               MIN(rating_score)                               AS min,
               MAX(rating_score)                               AS max,
               ROUND(PERCENTILE_APPROX(rating_score, 0.25), 4) AS q1,
               ROUND(PERCENTILE_APPROX(rating_score, 0.75), 4) AS q3
        FROM novels
        WHERE rating_score IS NOT NULL

        UNION ALL

        SELECT 'weighted_rating',
               COUNT(weighted_rating),
               ROUND(AVG(weighted_rating), 4),
               ROUND(PERCENTILE_APPROX(weighted_rating, 0.5), 4),
               ROUND(STDDEV(weighted_rating), 4),
               MIN(weighted_rating),
               MAX(weighted_rating),
               ROUND(PERCENTILE_APPROX(weighted_rating, 0.25), 4),
               ROUND(PERCENTILE_APPROX(weighted_rating, 0.75), 4)
        FROM novels
        WHERE weighted_rating IS NOT NULL

        UNION ALL

        SELECT 'chapter_count',
               COUNT(chapter_count),
               ROUND(AVG(chapter_count), 4),
               ROUND(PERCENTILE_APPROX(chapter_count, 0.5), 4),
               ROUND(STDDEV(chapter_count), 4),
               MIN(chapter_count),
               MAX(chapter_count),
               ROUND(PERCENTILE_APPROX(chapter_count, 0.25), 4),
               ROUND(PERCENTILE_APPROX(chapter_count, 0.75), 4)
        FROM novels
        WHERE chapter_count IS NOT NULL
        """
    )
    save(stats, "stat_descriptive")

    # Deteksi outlier metode IQR: batas = Q1 - 1.5*IQR dan Q3 + 1.5*IQR.
    # Dihitung di dalam SQL lewat CTE agar seluruh logikanya terdistribusi.
    outlier_summary = spark.sql(
        """
        WITH bounds AS (
            SELECT
                PERCENTILE_APPROX(chapter_count, 0.25) AS q1_ch,
                PERCENTILE_APPROX(chapter_count, 0.75) AS q3_ch,
                PERCENTILE_APPROX(rating_score, 0.25)  AS q1_rt,
                PERCENTILE_APPROX(rating_score, 0.75)  AS q3_rt
            FROM novels
        ),
        limits AS (
            SELECT
                q1_ch - 1.5 * (q3_ch - q1_ch) AS lo_ch,
                q3_ch + 1.5 * (q3_ch - q1_ch) AS hi_ch,
                q1_rt - 1.5 * (q3_rt - q1_rt) AS lo_rt,
                q3_rt + 1.5 * (q3_rt - q1_rt) AS hi_rt
            FROM bounds
        )
        SELECT 'chapter_count' AS metric,
               ROUND(l.lo_ch, 2) AS lower_bound,
               ROUND(l.hi_ch, 2) AS upper_bound,
               SUM(CASE WHEN n.chapter_count < l.lo_ch OR n.chapter_count > l.hi_ch
                        THEN 1 ELSE 0 END)                                AS n_outlier,
               COUNT(n.chapter_count)                                      AS n_total,
               ROUND(100.0 * SUM(CASE WHEN n.chapter_count < l.lo_ch OR n.chapter_count > l.hi_ch
                        THEN 1 ELSE 0 END) / COUNT(n.chapter_count), 2)    AS pct_outlier
        FROM novels n CROSS JOIN limits l
        WHERE n.chapter_count IS NOT NULL
        GROUP BY l.lo_ch, l.hi_ch

        UNION ALL

        SELECT 'rating_score',
               ROUND(l.lo_rt, 2), ROUND(l.hi_rt, 2),
               SUM(CASE WHEN n.rating_score < l.lo_rt OR n.rating_score > l.hi_rt THEN 1 ELSE 0 END),
               COUNT(n.rating_score),
               ROUND(100.0 * SUM(CASE WHEN n.rating_score < l.lo_rt OR n.rating_score > l.hi_rt
                        THEN 1 ELSE 0 END) / COUNT(n.rating_score), 2)
        FROM novels n CROSS JOIN limits l
        WHERE n.rating_score IS NOT NULL
        GROUP BY l.lo_rt, l.hi_rt
        """
    )
    save(outlier_summary, "stat_outlier_summary")

    # Daftar novel outlier terpanjang - berguna ditampilkan sebagai tabel di dashboard.
    outlier_rows = spark.sql(
        """
        WITH limits AS (
            SELECT PERCENTILE_APPROX(chapter_count, 0.75)
                   + 1.5 * (PERCENTILE_APPROX(chapter_count, 0.75)
                            - PERCENTILE_APPROX(chapter_count, 0.25)) AS hi_ch
            FROM novels
        )
        SELECT n.title, n.primary_genre, n.chapter_count, n.rating_score,
               n.rating_votes, n.weighted_rating
        FROM novels n CROSS JOIN limits l
        WHERE n.chapter_count > l.hi_ch
        ORDER BY n.chapter_count DESC
        LIMIT 50
        """
    )
    save(outlier_rows, "stat_outlier_novels", preview=5)


def analytic_2_categorical(spark: SparkSession) -> None:
    logger.info(">> Analitik 2: frekuensi genre & kombinasi genre")

    genre_freq = spark.sql(
        """
        SELECT genre,
               COUNT(*)                                     AS n_novels,
               ROUND(100.0 * COUNT(*) /
                     (SELECT COUNT(*) FROM novels), 2)      AS pct_of_catalog,
               ROUND(AVG(weighted_rating), 4)               AS avg_weighted_rating,
               ROUND(AVG(chapter_count), 1)                 AS avg_chapters
        FROM novel_genres
        GROUP BY genre
        ORDER BY n_novels DESC
        """
    )
    save(genre_freq, "genre_frequency")

    # Kombinasi genre: `genre_combo` sudah diurutkan alfabetis di Fase 2 sehingga
    # {Action, Romance} dan {Romance, Action} dihitung sebagai kombinasi sama.
    genre_combo = spark.sql(
        """
        SELECT genre_combo,
               n_genres,
               COUNT(*)                        AS n_novels,
               ROUND(AVG(weighted_rating), 4)  AS avg_weighted_rating
        FROM novels
        WHERE n_genres >= 2
        GROUP BY genre_combo, n_genres
        HAVING COUNT(*) >= 5
        ORDER BY n_novels DESC
        LIMIT 40
        """
    )
    save(genre_combo, "genre_combinations")

    # Pasangan genre (co-occurrence) - basis untuk heatmap keterkaitan genre.
    # Self-join dengan syarat a < b mencegah pasangan ganda dan pasangan-diri.
    genre_pairs = spark.sql(
        """
        SELECT a.genre AS genre_a,
               b.genre AS genre_b,
               COUNT(*) AS n_novels
        FROM novel_genres a
        JOIN novel_genres b
          ON a.series_id = b.series_id
         AND a.genre < b.genre
        GROUP BY a.genre, b.genre
        HAVING COUNT(*) >= 20
        ORDER BY n_novels DESC
        """
    )
    save(genre_pairs, "genre_pairs", preview=5)


def analytic_3_bivariate(spark: SparkSession) -> None:
    logger.info(">> Analitik 3: korelasi panjang novel vs rating")

    # Korelasi Pearson antar metrik numerik utama, dihitung pada populasi
    # ber-rating tepercaya agar tidak tertarik oleh novel bersuara 1-2.
    correlations = spark.sql(
        """
        SELECT 'chapter_count ~ weighted_rating' AS pair,
               ROUND(CORR(chapter_count, weighted_rating), 4) AS pearson_r,
               COUNT(*) AS n
        FROM novels
        WHERE is_rated_reliable AND chapter_count IS NOT NULL

        UNION ALL
        SELECT 'log_chapter_count ~ weighted_rating',
               ROUND(CORR(LN(chapter_count), weighted_rating), 4), COUNT(*)
        FROM novels
        WHERE is_rated_reliable AND chapter_count IS NOT NULL AND chapter_count > 0

        UNION ALL
        SELECT 'rating_votes ~ weighted_rating',
               ROUND(CORR(rating_votes, weighted_rating), 4), COUNT(*)
        FROM novels
        WHERE is_rated_reliable

        UNION ALL
        SELECT 'reading_list_count ~ weighted_rating',
               ROUND(CORR(reading_list_count, weighted_rating), 4), COUNT(reading_list_count)
        FROM novels
        WHERE is_rated_reliable

        UNION ALL
        SELECT 'chapter_count ~ rating_votes',
               ROUND(CORR(chapter_count, rating_votes), 4), COUNT(*)
        FROM novels
        WHERE is_rated_reliable AND chapter_count IS NOT NULL

        -- Kecepatan rilis, diurai dari `release_frequency` di Fase 2. Kolom itu
        -- sebelumnya hanya lewat sebagai teks; ternyata 100% dapat diurai dan
        -- korelasinya dengan rating sekuat korelasi jumlah bab.
        -- Di-log karena sebarannya sangat miring (median 37 hari, maks 20.610).
        UNION ALL
        SELECT 'log_days_per_chapter ~ weighted_rating',
               ROUND(CORR(LN(days_per_chapter), weighted_rating), 4), COUNT(*)
        FROM novels
        WHERE is_rated_reliable AND days_per_chapter IS NOT NULL AND days_per_chapter > 0
        """
    )
    save(correlations, "correlation_metrics")

    # Kecepatan rilis per kelompok — memperlihatkan bentuk hubungannya, yang
    # tidak terlihat dari satu angka korelasi saja.
    release_vs_rating = spark.sql(
        """
        SELECT release_pace,
               COUNT(*)                                          AS n_novels,
               ROUND(AVG(weighted_rating), 4)                    AS avg_weighted_rating,
               ROUND(PERCENTILE_APPROX(weighted_rating, 0.5), 4) AS median_weighted_rating,
               ROUND(AVG(days_per_chapter), 1)                   AS avg_days_per_chapter,
               ROUND(AVG(chapter_count), 1)                      AS avg_chapters
        FROM novels
        WHERE release_pace IS NOT NULL AND is_rated_reliable
        GROUP BY release_pace
        ORDER BY release_pace
        """
    )
    save(release_vs_rating, "release_pace_vs_rating")

    # Korelasi tunggal bisa menyembunyikan hubungan non-linear, jadi hubungan
    # yang sama juga dilihat per bucket panjang novel.
    length_vs_rating = spark.sql(
        """
        SELECT length_bucket,
               COUNT(*)                                   AS n_novels,
               ROUND(AVG(weighted_rating), 4)             AS avg_weighted_rating,
               ROUND(PERCENTILE_APPROX(weighted_rating, 0.5), 4) AS median_weighted_rating,
               ROUND(AVG(rating_votes), 1)                AS avg_votes,
               ROUND(AVG(chapter_count), 1)               AS avg_chapters
        FROM novels
        WHERE length_bucket IS NOT NULL AND is_rated_reliable
        GROUP BY length_bucket
        ORDER BY length_bucket
        """
    )
    save(length_vs_rating, "length_vs_rating")


def analytic_4_multidim(spark: SparkSession) -> None:
    logger.info(">> Analitik 4: peringkat genre & agregasi multi-dimensi")

    genre_ranking = spark.sql(
        f"""
        SELECT genre,
               COUNT(*)                                          AS n_novels,
               ROUND(AVG(weighted_rating), 4)                    AS avg_weighted_rating,
               ROUND(AVG(chapter_count), 1)                      AS avg_chapters,
               ROUND(PERCENTILE_APPROX(chapter_count, 0.5), 1)   AS median_chapters,
               RANK() OVER (ORDER BY AVG(weighted_rating) DESC)  AS rank_by_rating,
               RANK() OVER (ORDER BY AVG(chapter_count) DESC)    AS rank_by_length
        FROM novel_genres
        WHERE is_rated_reliable
        GROUP BY genre
        HAVING COUNT(*) >= {MIN_GROUP_SUPPORT}
        ORDER BY avg_weighted_rating DESC
        """
    )
    save(genre_ranking, "genre_ranking")

    # Agregasi 2 dimensi: genre x negara asal. Ini yang menjawab pertanyaan
    # bisnis "genre apa yang unggul di pasar mana", bukan sekadar genre global.
    genre_origin = spark.sql(
        f"""
        SELECT genre,
               type_origin,
               COUNT(*)                       AS n_novels,
               ROUND(AVG(weighted_rating), 4) AS avg_weighted_rating,
               ROUND(AVG(chapter_count), 1)   AS avg_chapters
        FROM novel_genres
        WHERE is_rated_reliable AND type_origin IS NOT NULL
        GROUP BY genre, type_origin
        HAVING COUNT(*) >= {MIN_GROUP_SUPPORT}
        ORDER BY avg_weighted_rating DESC
        """
    )
    save(genre_origin, "genre_origin_matrix")

    # Agregasi lintas dimensi dengan GROUPING SETS: satu kueri memberi ringkasan per asal, per tipe, dan kombinasinya.
    # Label 'ALL' hanya untuk subtotal (GROUPING() = 1); NULL asli diberi label 'Unknown' agar tidak terbaca sebagai grand total.
    origin_type = spark.sql(
        """
        SELECT CASE WHEN GROUPING(type_origin) = 1 THEN 'ALL'
                    ELSE COALESCE(type_origin, 'Unknown') END AS type_origin,
               CASE WHEN GROUPING(type) = 1 THEN 'ALL'
                    ELSE COALESCE(type, 'Unknown') END        AS type,
               COUNT(*)                      AS n_novels,
               ROUND(AVG(weighted_rating), 4) AS avg_weighted_rating,
               ROUND(AVG(chapter_count), 1)  AS avg_chapters,
               ROUND(AVG(popularity_score), 3) AS avg_popularity
        FROM novels
        WHERE is_rated_reliable
        GROUP BY GROUPING SETS ((type_origin), (type), (type_origin, type))
        HAVING COUNT(*) >= 20
        ORDER BY n_novels DESC
        """
    )
    save(origin_type, "origin_type_summary", preview=6)

    # Tren temporal: apakah novel yang lebih baru dinilai lebih tinggi?
    decade_trend = spark.sql(
        """
        SELECT decade,
               COUNT(*)                       AS n_novels,
               ROUND(AVG(weighted_rating), 4) AS avg_weighted_rating,
               ROUND(AVG(chapter_count), 1)   AS avg_chapters
        FROM novels
        WHERE decade IS NOT NULL AND is_rated_reliable
        GROUP BY decade
        HAVING COUNT(*) >= 10
        ORDER BY decade
        """
    )
    save(decade_trend, "decade_trend", preview=6)


def analytic_5_textual(spark: SparkSession) -> None:
    logger.info(">> Analitik 5: metrik teks dari sinopsis & tag")

    # Frekuensi token sinopsis untuk word cloud Fase 5; korpus gabungan hanya akan menampilkan frasa tag ("male protagonist").
    top_tokens = spark.sql(
        """
        SELECT token,
               COUNT(*)                     AS freq,
               COUNT(DISTINCT series_id)    AS n_novels
        FROM novel_synopsis_tokens
        GROUP BY token
        HAVING COUNT(*) >= 5
        ORDER BY freq DESC
        LIMIT 300
        """
    )
    save(top_tokens, "top_tokens", preview=10)

    # Tag utuh (frasa) lebih informatif dari token tunggal untuk analisis bisnis.
    top_tags = spark.sql(
        f"""
        SELECT tag,
               COUNT(*)                       AS n_novels,
               ROUND(AVG(weighted_rating), 4) AS avg_weighted_rating
        FROM novel_tags
        GROUP BY tag
        HAVING COUNT(*) >= {MIN_GROUP_SUPPORT}
        ORDER BY n_novels DESC
        LIMIT 100
        """
    )
    save(top_tags, "top_tags", preview=8)

    # Tag dengan rating rata-rata tertinggi - "elemen cerita apa yang disukai
    # pembaca", pertanyaan yang tidak terjawab oleh frekuensi semata.
    best_tags = spark.sql(
        f"""
        SELECT tag,
               COUNT(*)                       AS n_novels,
               ROUND(AVG(weighted_rating), 4) AS avg_weighted_rating
        FROM novel_tags
        WHERE is_rated_reliable
        GROUP BY tag
        HAVING COUNT(*) >= {MIN_GROUP_SUPPORT}
        ORDER BY avg_weighted_rating DESC
        LIMIT 30
        """
    )
    save(best_tags, "best_rated_tags", preview=8)

    # Poin analitik inti: hubungan PANJANG SINOPSIS dengan rating.
    # Apakah novel yang sinopsisnya lebih panjang/rinci dinilai lebih tinggi?
    text_vs_rating = spark.sql(
        """
        SELECT CASE
                   WHEN synopsis_word_count <  50 THEN '1. Sangat ringkas (<50 kata)'
                   WHEN synopsis_word_count < 100 THEN '2. Ringkas (50-99)'
                   WHEN synopsis_word_count < 175 THEN '3. Sedang (100-174)'
                   WHEN synopsis_word_count < 300 THEN '4. Panjang (175-299)'
                   ELSE '5. Sangat panjang (300+)'
               END                                 AS text_richness,
               COUNT(*)                            AS n_novels,
               ROUND(AVG(synopsis_word_count), 1)  AS avg_tokens,
               ROUND(AVG(weighted_rating), 4)      AS avg_weighted_rating,
               ROUND(AVG(chapter_count), 1)        AS avg_chapters,
               ROUND(AVG(rating_votes), 1)         AS avg_votes
        FROM novels
        WHERE is_rated_reliable AND has_synopsis
        GROUP BY 1
        ORDER BY 1
        """
    )
    save(text_vs_rating, "text_richness_vs_rating")

    # Kata sinopsis pembeda novel disukai dan biasa: frekuensi token pada kuartil rating teratas vs terbawah.
    # Dihitung per novel (DISTINCT series_id, token) agar nama tokoh yang diulang 8x tidak terhitung 8 novel.
    discriminative = spark.sql(
        """
        WITH doc_tokens AS (
            SELECT DISTINCT series_id, token FROM novel_synopsis_tokens
        ),
        bounds AS (
            SELECT PERCENTILE_APPROX(weighted_rating, 0.75) AS hi,
                   PERCENTILE_APPROX(weighted_rating, 0.25) AS lo
            FROM novels WHERE is_rated_reliable AND has_synopsis
        ),
        labelled AS (
            SELECT n.series_id,
                   CASE WHEN n.weighted_rating >= b.hi THEN 'top'
                        WHEN n.weighted_rating <= b.lo THEN 'bottom' END AS bucket
            FROM novels n CROSS JOIN bounds b
            WHERE n.is_rated_reliable AND n.has_synopsis
              AND (n.weighted_rating >= b.hi OR n.weighted_rating <= b.lo)
        )
        SELECT t.token,
               SUM(CASE WHEN l.bucket = 'top'    THEN 1 ELSE 0 END) AS n_top,
               SUM(CASE WHEN l.bucket = 'bottom' THEN 1 ELSE 0 END) AS n_bottom,
               ROUND(
                   (SUM(CASE WHEN l.bucket = 'top' THEN 1 ELSE 0 END) + 1.0) /
                   (SUM(CASE WHEN l.bucket = 'bottom' THEN 1 ELSE 0 END) + 1.0), 3
               ) AS top_to_bottom_ratio
        FROM doc_tokens t
        JOIN labelled l ON t.series_id = l.series_id
        GROUP BY t.token
        HAVING COUNT(*) >= 40
        ORDER BY top_to_bottom_ratio DESC
        LIMIT 40
        """
    )
    save(discriminative, "synopsis_discriminative_tokens", preview=8)

    text_corr = spark.sql(
        """
        SELECT 'synopsis_word_count ~ weighted_rating' AS pair,
               ROUND(CORR(synopsis_word_count, weighted_rating), 4) AS pearson_r, COUNT(*) AS n
        FROM novels WHERE is_rated_reliable AND has_synopsis
        UNION ALL
        SELECT 'synopsis_word_count ~ rating_votes',
               ROUND(CORR(synopsis_word_count, rating_votes), 4), COUNT(*)
        FROM novels WHERE is_rated_reliable AND has_synopsis
        UNION ALL
        SELECT 'token_count ~ weighted_rating',
               ROUND(CORR(token_count, weighted_rating), 4), COUNT(*)
        FROM novels WHERE is_rated_reliable
        UNION ALL
        SELECT 'n_tags ~ weighted_rating',
               ROUND(CORR(n_tags, weighted_rating), 4), COUNT(*)
        FROM novels WHERE is_rated_reliable
        UNION ALL
        SELECT 'n_tags ~ rating_votes',
               ROUND(CORR(n_tags, rating_votes), 4), COUNT(*)
        FROM novels WHERE is_rated_reliable
        UNION ALL
        SELECT 'synopsis_word_count ~ chapter_count',
               ROUND(CORR(synopsis_word_count, chapter_count), 4), COUNT(*)
        FROM novels WHERE chapter_count IS NOT NULL AND has_synopsis
        """
    )
    save(text_corr, "text_correlation")


def export_viz_dataset(spark: SparkSession) -> None:
    """
    Ekspor tabel per-novel yang ringkas untuk scatter plot & boxplot.

    Kolom teks berat (tokens, corpus) sengaja tidak diikutkan agar dashboard
    tidak perlu menarik payload besar dari object storage.
    """
    logger.info(">> Ekspor dataset visualisasi per-novel")
    viz = spark.sql(
        """
        SELECT series_id, title, url, primary_genre, genre_combo, n_genres, n_tags,
               type, type_origin, language, year, decade, status_coo,
               chapter_count, length_bucket, is_completed,
               rating_score, rating_votes, weighted_rating, is_rated_reliable,
               reading_list_count, popularity_score, token_count,
               synopsis_word_count, has_synopsis,
               days_per_chapter, release_pace
        FROM novels
        """
    )
    save(viz, "viz_novels", preview=0)


def run() -> None:
    log_section(logger, "fase 3 - explore (spark sql analytics)")
    with spark_session("phase3-explore") as spark:
        register_views(spark)
        analytic_1_descriptive(spark)
        analytic_2_categorical(spark)
        analytic_3_bivariate(spark)
        analytic_4_multidim(spark)
        analytic_5_textual(spark)
        export_viz_dataset(spark)
    logger.info("FASE 3 SELESAI - seluruh hasil analitik tersimpan di curated-data/analytics/")


def main() -> int:
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        logger.error("FASE 3 GAGAL: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
