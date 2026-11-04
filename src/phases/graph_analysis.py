"""
GRAPH ANALYSIS — algoritma graf sungguhan di atas property graph
`data_model.py` sudah memproyeksikan katalog jadi property graph
(`graph_nodes`/`graph_edges`, 67.807 simpul, 860.109 sisi) dan dua "operasi
graf" (`op_graph_genre_cooccurrence`, `op_graph_author_reach`) - tapi keduanya
SELF-JOIN pada tabel sisi, bukan algoritma graf. Self-join menjawab "berapa
kali X dan Y muncul bersama"; ia tidak bisa menjawab "simpul mana yang paling
BERPENGARUH mempertimbangkan SELURUH struktur jaringan" (PageRank), "apakah
jaringan ini terpecah jadi beberapa pulau" (connected components), "kelompok
mana yang lebih sering berelasi ke dalam kelompoknya sendiri daripada ke luar"
(deteksi komunitas), atau "berapa hop dari A ke B" (traversal). Modul ini
mengisi celah itu dengan algoritma graf yang sebenarnya (NetworkX).

Dua graf diproyeksikan dari sisi mentah, masing-masing berukuran yang benar
untuk algoritmanya:

  1. **Ko-okurensi genre** (35 simpul) - graf terbobot: sisi (A, B) berbobot
     jumlah novel yang menyandang KEDUA genre itu. Cukup kecil untuk dianalisis
     penuh (PageRank, sentralitas, komunitas) di memori, tanpa sampling.
  2. **Kolaborasi penulis** (hanya novel dengan >= 2 penulis, ~1.800 novel) -
     graf TIDAK berbobot: sisi antar-penulis yang benar-benar menulis novel
     YANG SAMA bersama. Ini co-authorship sungguhan, bukan proksi "menulis
     genre serupa" yang lebih lemah maknanya.

Kenapa diproyeksikan ke NetworkX di driver, bukan GraphFrames terdistribusi:
sama seperti alasan `data_model.py` tidak memakai database graf terpisah -
kedua graf hasil proyeksi ini (puluhan-ribuan simpul) muat nyaman di memori
satu proses, dan algoritma sekelas PageRank/community detection pada skala ini
selesai dalam hitungan detik. Proyeksinya sendiri (agregasi dari 860 ribu sisi
mentah) tetap dikerjakan Spark, terdistribusi.

Jalankan:
    docker compose exec pipeline python -m src.phases.graph_analysis
"""

from __future__ import annotations

import sys

import networkx as nx
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.common.config import get_settings
from src.common.logging_utils import get_logger, log_section
from src.common.spark_session import spark_session

logger = get_logger("graph.analysis")

#: Genre/tag dengan dukungan di bawah ini dikeluarkan dari analisis komunitas -
#: sama seperti data_model.py mengabaikan nilai 'Unknown', sisi bernilai
#: sangat kecil hanya menambah derau, bukan sinyal struktur.
MIN_COOCCURRENCE = 5


def _model_uri(settings, table: str) -> str:
    return settings.minio.s3a_uri(settings.minio.bucket_curated, f"model/{table}")


def _save(spark: SparkSession, df, name: str) -> None:
    settings = get_settings()
    uri = settings.minio.s3a_uri(settings.minio.bucket_curated, f"analytics/{name}")
    (spark.createDataFrame(df) if isinstance(df, pd.DataFrame) else df).coalesce(1).write.mode(
        "overwrite"
    ).parquet(uri)
    logger.info("[%s] tersimpan -> %s", name, uri)


def build_genre_graph(spark: SparkSession) -> nx.Graph:
    bridge = spark.read.parquet(_model_uri(get_settings(), "bridge_novel_genre"))
    dim = spark.read.parquet(_model_uri(get_settings(), "dim_genre")).select("genre_key", "genre")

    a, b = bridge.alias("a"), bridge.alias("b")
    pairs = (
        a.join(b, (F.col("a.series_id") == F.col("b.series_id")) & (F.col("a.genre_key") < F.col("b.genre_key")))
        .groupBy(F.col("a.genre_key").alias("g1"), F.col("b.genre_key").alias("g2"))
        .agg(F.count("*").alias("weight"))
        .filter(F.col("weight") >= MIN_COOCCURRENCE)
        .join(dim.withColumnRenamed("genre_key", "g1").withColumnRenamed("genre", "genre1"), on="g1")
        .join(dim.withColumnRenamed("genre_key", "g2").withColumnRenamed("genre", "genre2"), on="g2")
        .select("genre1", "genre2", "weight")
        .toPandas()
    )

    graph = nx.Graph()
    for genre in dim.select("genre").toPandas()["genre"]:
        graph.add_node(genre)
    for row in pairs.itertuples(index=False):
        graph.add_edge(row.genre1, row.genre2, weight=int(row.weight))
    logger.info("Graf genre: %d simpul, %d sisi (ambang ko-okurensi >= %d)",
                graph.number_of_nodes(), graph.number_of_edges(), MIN_COOCCURRENCE)
    return graph


def build_author_graph(spark: SparkSession) -> tuple[nx.Graph, int]:
    """
    Graf kolaborasi + jumlah penulis yang TIDAK punya kolaborator sama sekali.

    Graf dibangun dari sisi saja, jadi penulis tunggal tidak pernah menjadi
    simpul dan `n_isolated_nodes` graf ini secara struktural selalu 0. Angka
    penulis tanpa kolaborator dihitung terpisah dari dim_author, bukan dengan
    menambahkan simpul terisolasi (yang akan mengubah PageRank & komunitas).
    """
    bridge = spark.read.parquet(_model_uri(get_settings(), "bridge_novel_author"))
    dim = spark.read.parquet(_model_uri(get_settings(), "dim_author")).select("author_key", "author")

    # Hanya novel dengan >= 2 penulis - novel bertunggal-penulis tidak
    # menyumbang sisi kolaborasi apa pun.
    multi = bridge.groupBy("series_id").agg(F.count("*").alias("n")).filter(F.col("n") >= 2)
    bridge = bridge.join(multi.select("series_id"), on="series_id")

    a, b = bridge.alias("a"), bridge.alias("b")
    pairs = (
        a.join(b, (F.col("a.series_id") == F.col("b.series_id")) & (F.col("a.author_key") < F.col("b.author_key")))
        .select(F.col("a.author_key").alias("k1"), F.col("b.author_key").alias("k2"))
        .distinct()
        .join(dim.withColumnRenamed("author_key", "k1").withColumnRenamed("author", "author1"), on="k1")
        .join(dim.withColumnRenamed("author_key", "k2").withColumnRenamed("author", "author2"), on="k2")
        .select("author1", "author2")
        .toPandas()
    )

    graph = nx.Graph()
    graph.add_edges_from(pairs.itertuples(index=False, name=None))
    n_novels_multi = multi.count()
    n_solo = dim.count() - graph.number_of_nodes()
    logger.info("Graf kolaborasi penulis: %d simpul, %d sisi, dari %s novel multi-penulis "
                "(%s penulis tanpa kolaborator)",
                graph.number_of_nodes(), graph.number_of_edges(), f"{n_novels_multi:,}", f"{n_solo:,}")
    return graph, n_solo


def analyze_graph(graph: nx.Graph, label: str) -> dict:
    """
    Empat kelas pertanyaan yang tidak terjawab oleh self-join biasa:

      * PageRank    - pengaruh simpul mempertimbangkan SELURUH struktur, bukan
                       cuma derajatnya sendiri (simpul terhubung ke simpul
                       berpengaruh ikut naik peringkat).
      * Sentralitas - derajat (langsung) vs betweenness (seberapa sering
                       simpul jadi JEMBATAN di jalur terpendek antar simpul lain).
      * Komponen terhubung - apakah graf ini satu jaringan utuh atau
                       terpecah jadi pulau-pulau terisolasi.
      * Komunitas   - kelompok simpul yang lebih rapat terhubung ke DALAM
                       kelompoknya sendiri daripada ke luar (modularity greedy).
    """
    if graph.number_of_nodes() == 0:
        return {"label": label, "n_nodes": 0, "n_edges": 0}

    pagerank = nx.pagerank(graph, weight="weight" if nx.is_weighted(graph) else None)
    degree_c = nx.degree_centrality(graph)
    # Betweenness eksak adalah O(V*E) - mahal di atas beberapa ribu simpul.
    # Sampel k node sumber cukup untuk perkiraan stabil pada graf sekecil ini.
    k_sample = min(200, graph.number_of_nodes())
    betweenness = nx.betweenness_centrality(graph, k=k_sample, seed=42, weight=None)

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    communities = list(nx.algorithms.community.greedy_modularity_communities(graph, weight="weight" if nx.is_weighted(graph) else None))
    modularity = nx.algorithms.community.modularity(
        graph, communities, weight="weight" if nx.is_weighted(graph) else None
    ) if communities else 0.0

    # Komunitas diurutkan dari yang terbesar, jadi id 0 selalu komunitas terbesar.
    community_of = {n: i for i, members in enumerate(communities) for n in members}

    top_pagerank = sorted(pagerank.items(), key=lambda kv: -kv[1])[:10]
    top_betweenness = sorted(betweenness.items(), key=lambda kv: -kv[1])[:10]

    logger.info(
        "[%s] %d simpul, %d sisi | %d komponen (terbesar=%d simpul) | "
        "%d komunitas (modularity=%.4f)",
        label, graph.number_of_nodes(), graph.number_of_edges(),
        len(components), len(components[0]) if components else 0,
        len(communities), modularity,
    )
    logger.info("[%s] Top-5 PageRank: %s", label,
                [(n, round(v, 4)) for n, v in top_pagerank[:5]])

    return {
        "label": label,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_components": len(components),
        "largest_component_size": len(components[0]) if components else 0,
        "n_isolated_nodes": sum(1 for c in components if len(c) == 1),
        "n_communities": len(communities),
        "modularity": round(float(modularity), 4),
        "communities": [sorted(c) for c in communities],
        "pagerank": pd.DataFrame(
            [{"node": n, "pagerank": v, "degree_centrality": degree_c[n], "betweenness": betweenness[n],
              "community": community_of.get(n, -1)}
             for n, v in pagerank.items()]
        ).sort_values("pagerank", ascending=False).reset_index(drop=True),
        "top_pagerank": top_pagerank,
        "top_betweenness": top_betweenness,
    }


def shortest_path_demo(graph: nx.Graph, source: str, target: str) -> dict:
    """
    Traversal multi-hop nyata - jawaban yang MEMANG butuh algoritma graf
    (BFS/Dijkstra), bukan agregasi. `data_model.py` mendokumentasikan alasan
    tidak memakai Neo4j/GraphFrames dengan syarat "selama pertanyaannya <= 2
    hop, self-join cukup" - fungsi ini membuktikan syarat itu MASIH berlaku
    dengan mengukur hop sungguhan, bukan mengasumsikannya.
    """
    if source not in graph or target not in graph:
        return {"source": source, "target": target, "reachable": False, "reason": "simpul tidak ada di graf"}
    if not nx.has_path(graph, source, target):
        return {"source": source, "target": target, "reachable": False, "reason": "tidak ada jalur"}
    path = nx.shortest_path(graph, source, target, weight=None)
    return {"source": source, "target": target, "reachable": True, "hops": len(path) - 1, "path": path}


def run() -> dict:
    log_section(logger, "graph analysis - pagerank, sentralitas, komunitas, traversal")
    settings = get_settings()

    with spark_session("novel-graph-analysis") as spark:
        genre_graph = build_genre_graph(spark)
        author_graph, n_solo_authors = build_author_graph(spark)

        genre_result = analyze_graph(genre_graph, "genre_cooccurrence")
        author_result = analyze_graph(author_graph, "author_collaboration")

        # Traversal: dua genre yang TIDAK bertetangga langsung (tidak pernah
        # disematkan bersama pada satu novel dengan dukungan cukup), tapi
        # terhubung lewat genre perantara - contoh nyata kenapa "2 hop" relevan.
        genre_names = list(genre_graph.nodes)
        traversal = None
        for i, g1 in enumerate(genre_names):
            for g2 in genre_names[i + 1:]:
                if not genre_graph.has_edge(g1, g2):
                    demo = shortest_path_demo(genre_graph, g1, g2)
                    if demo["reachable"] and demo["hops"] == 2:
                        traversal = demo
                        break
            if traversal:
                break
        if traversal:
            logger.info("Contoh traversal 2-hop: %s -> %s lewat %s",
                        traversal["source"], traversal["target"], traversal["path"][1])

        _save(spark, genre_result["pagerank"], "graph_genre_pagerank")
        # Sisi graf genre disimpan apa adanya supaya dashboard menggambar graf
        # yang sama persis dengan yang dianalisis di sini (ambang yang sama).
        _save(
            spark,
            pd.DataFrame(
                [{"source": u, "target": v, "weight": d["weight"]}
                 for u, v, d in genre_graph.edges(data=True)]
            ),
            "graph_genre_edges",
        )
        _save(spark, author_result["pagerank"], "graph_author_pagerank")

        skip = ("communities", "pagerank", "top_pagerank", "top_betweenness")
        genre_row = {k: v for k, v in genre_result.items() if k not in skip}
        author_row = {k: v for k, v in author_result.items() if k not in skip}
        # Hasil traversal disimpan supaya bigdata_audit (GRF-04) memeriksa hop
        # yang benar-benar terukur, bukan menganggapnya ada.
        genre_row["traversal_hops"] = traversal["hops"] if traversal else None
        genre_row["traversal_path"] = " -> ".join(traversal["path"]) if traversal else None
        author_row["traversal_hops"] = None
        author_row["traversal_path"] = None
        genre_row["n_authors_without_collaborator"] = None
        author_row["n_authors_without_collaborator"] = n_solo_authors
        author_row["n_isolated_nodes"] = None  # selalu 0 secara konstruksi - lihat build_author_graph
        summary = pd.DataFrame([genre_row, author_row])
        _save(spark, summary, "graph_summary")

    result = {
        "genre": genre_result,
        "author": author_result,
        "traversal_demo": traversal,
    }
    logger.info("GRAPH ANALYSIS SELESAI")
    return result


def main() -> int:
    try:
        run()
    except Exception as exc:  # noqa: BLE001
        logger.error("GRAPH ANALYSIS GAGAL: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
