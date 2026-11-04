"""
Graf kemiripan novel - graph analytics di atas representasi model
`graph_analysis.py` menganalisis graf yang sisinya berasal dari metadata
(genre yang disematkan bersama, penulis yang menulis bersama). Graf di modul
ini dibangun dari keluaran pipeline machine learning: setiap novel dihubungkan
ke 10 tetangga terdekatnya di ruang TF-IDF -> SVD 200 dimensi yang juga dipakai
Head B (rekomendasi) dan Head C (klaster arketipe).

Pertanyaan yang dijawab:

  * Novel mana yang menjadi pusat ruang konten (PageRank pada graf kNN)?
  * Apakah komunitas Louvain di graf ini sejalan dengan klaster K-Means Head C
    dan dengan genre utama (NMI / ARI)?
  * Genre mana yang novelnya berkumpul di ruang fitur (homofili), dan apakah
    homofili itu menjelaskan mengapa sebagian genre sulit diprediksi Head A?

Masukan hanya artefak lokal Fase 4 (`Model/artifacts/embeddings.npy` dan
`catalog.parquet`, urutan barisnya sama), sehingga modul ini tidak butuh Spark.
Hasilnya ditulis ke `Doc/generated/` (laporan) dan ke zona curated
`analytics/graph_similarity_*` (dibaca halaman Content Graph di dashboard).

Jalankan di container pipeline, supaya versi networkx sama dengan run resmi:
    docker compose exec pipeline python -m src.phases.similarity_graph
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.common.logging_utils import get_logger

logger = get_logger("graph.similarity")

PROJECT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = PROJECT_DIR / "Model/artifacts"
REPORT_PATH = PROJECT_DIR / "Doc/generated/similarity_graph.json"
EDGES_PATH = PROJECT_DIR / "Doc/generated/similarity_graph_edges.parquet"
NODES_PATH = PROJECT_DIR / "Doc/generated/similarity_graph_nodes.parquet"

K_NEIGHBOURS = 10
SEED = 42
# Head A yang dipakai untuk mengaitkan homofili dengan F1 per genre: model bawaan dashboard
# (best_model di metrics.json), karena prediksi model itulah yang dikonfirmasi kurator.
FALLBACK_REFERENCE_MODEL = "svm"


def knn_edges(embeddings: np.ndarray, k: int = K_NEIGHBOURS, chunk: int = 1024, workers: int = 4) -> pd.DataFrame:
    """
    Sisi kNN berbobot cosine. Embedding sudah ternormalisasi L2, jadi hasil kali
    titik sama dengan cosine. Dihitung per potongan baris supaya matriks
    kemiripan penuh (39 ribu x 39 ribu) tidak pernah ada di memori.

    Potongan dikerjakan paralel di thread: matmul dan argpartition NumPy melepas
    GIL, jadi 4 thread memangkas ~21 dtk menjadi ~10 dtk pada 39.660 novel dengan
    puncak memori ~1,3 GB. Seleksi tetap `argpartition(-block, k)` supaya pemecah
    seri pada kemiripan kembar (sinopsis duplikat) sama dengan run yang dilaporkan.
    """
    def neighbours(start: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        block = embeddings[start:start + chunk] @ embeddings.T
        rows = np.arange(block.shape[0])
        block[rows, start + rows] = -np.inf
        top = np.argpartition(-block, k, axis=1)[:, :k]
        return np.repeat(start + rows, k), top.ravel(), np.take_along_axis(block, top, axis=1).ravel()

    with ThreadPoolExecutor(max_workers=max(1, min(workers, os.cpu_count() or 1))) as pool:
        parts = list(pool.map(neighbours, range(0, len(embeddings), chunk)))
    src, dst, weight = (np.concatenate(column) for column in zip(*parts))
    # Graf tak berarah: pasangan (a, b) dan (b, a) digabung, bobot diambil yang terbesar.
    edges = pd.DataFrame({"src": np.minimum(src, dst), "dst": np.maximum(src, dst), "weight": weight})
    return edges.groupby(["src", "dst"], as_index=False)["weight"].max()


def build_graph(edges: pd.DataFrame, n_nodes: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_weighted_edges_from(edges.itertuples(index=False, name=None))
    return graph


def genre_homophily(graph: nx.Graph, genre_sets: list[set], labels: list[str]) -> pd.DataFrame:
    """
    Homofili per genre: dari seluruh tetangga novel bergenre G, berapa persen
    yang juga bergenre G, dibandingkan dengan prevalensi G di katalog.
    Lift > 1 berarti novel G saling berdekatan di ruang fitur.
    """
    n = len(genre_sets)
    rows = []
    for genre in labels:
        members = [i for i, s in enumerate(genre_sets) if genre in s]
        prevalence = len(members) / n
        same = total = 0
        for node in members:
            for nb in graph.neighbors(node):
                total += 1
                same += genre in genre_sets[nb]
        share = same / total if total else 0.0
        rows.append({"genre": genre, "n_novels": len(members), "prevalence": prevalence,
                     "neighbour_share": share, "lift": share / prevalence if prevalence else 0.0})
    return pd.DataFrame(rows)


def community_profiles(catalog: pd.DataFrame, community: np.ndarray, top: int | None = 8) -> list[dict]:
    profiles = []
    for cid, size in Counter(community).most_common(top):
        members = catalog[community == cid]
        genres = Counter(g for row in members["genre_list"] for g in (row if row is not None else []))
        origins = members["type_origin"].fillna("tidak diketahui").value_counts(normalize=True)
        profiles.append({
            "community": int(cid),
            "size": int(size),
            "top_genres": [[g, c] for g, c in genres.most_common(4)],
            "top_origin": [origins.index[0], round(float(origins.iloc[0]), 3)],
            "avg_weighted_rating": round(float(members["weighted_rating"].mean()), 3),
            "example_titles": members.nlargest(3, "rating_votes")["title"].tolist(),
        })
    return profiles


def partial_spearman(y: pd.Series, x: pd.Series, control: pd.Series) -> float:
    """Korelasi Spearman y dan x setelah pengaruh `control` dikeluarkan dari keduanya (residual peringkat)."""
    def resid(v):
        design = np.c_[np.ones(len(v)), rankdata(control)]
        coef = np.linalg.lstsq(design, rankdata(v), rcond=None)[0]
        return rankdata(v) - design @ coef
    return float(np.corrcoef(resid(y), resid(x))[0, 1])


#: Genre yang ada di hampir semua komunitas; label komunitas memakai genre berikutnya agar pembeda terlihat.
UBIQUITOUS = {"Romance", "Fantasy", "Drama"}


def lake_tables(catalog: pd.DataFrame, edges: pd.DataFrame, community: np.ndarray, homophily: pd.DataFrame,
                report: dict) -> dict[str, pd.DataFrame]:
    """Tabel untuk dashboard; kunci simpul diganti `series_id` supaya bisa digabung dengan katalog."""
    ids = catalog["series_id"].to_numpy()
    rows = []
    for p in community_profiles(catalog, community, top=None):
        genres = [g for g, _ in p["top_genres"]]
        distinctive = [g for g in genres if g not in UBIQUITOUS][:2] or genres[:2]
        rows.append({
            "community": p["community"], "size": p["size"], "label": " + ".join(distinctive),
            "top_genres": ", ".join(genres), "top_origin": p["top_origin"][0], "origin_share": p["top_origin"][1],
            "avg_weighted_rating": p["avg_weighted_rating"], "example_titles": " | ".join(p["example_titles"]),
        })
    pair = pd.DataFrame({"a": community[edges["src"].to_numpy()], "b": community[edges["dst"].to_numpy()]})
    pair = pair[pair["a"] != pair["b"]]
    pair = pair.assign(lo=pair[["a", "b"]].min(axis=1), hi=pair[["a", "b"]].max(axis=1))
    community_edges = pair.groupby(["lo", "hi"]).size().rename("n_edges").reset_index()

    summary = {k: v for k, v in report.items() if not isinstance(v, (dict, list))}
    summary.update({f"degree_{k}": v for k, v in report["degree"].items()})
    summary.update(report["agreement"])
    summary.update({f"spearman_f1_{k}": v[0] for k, v in report["spearman_f1"].items()})
    return {
        "graph_similarity_summary": pd.DataFrame([summary]),
        "graph_similarity_homophily": homophily,
        "graph_similarity_communities": pd.DataFrame(rows),
        "graph_similarity_community_edges": community_edges,
        "graph_similarity_nodes": catalog[["series_id", "title", "primary_genre", "community", "pagerank", "degree",
                                           "rating_votes", "weighted_rating"]],
        "graph_similarity_edges": pd.DataFrame({"src": ids[edges["src"].to_numpy()], "dst": ids[edges["dst"].to_numpy()],
                                                "weight": edges["weight"].astype("float32")}),
    }


def publish(tables: dict[str, pd.DataFrame]) -> None:
    """Satu berkas parquet per tabel di `curated-data/analytics/<nama>/`, format yang dibaca read_parquet_dir."""
    from src.common.config import get_settings
    from src.common.storage import upload_file

    settings = get_settings()
    with tempfile.TemporaryDirectory() as tmp:
        for name, frame in tables.items():
            path = Path(tmp) / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            upload_file(path, settings.minio.bucket_curated, f"analytics/{name}/part-00000.parquet", settings)


def run(to_lake: bool = True) -> dict:
    catalog = pd.read_parquet(ARTIFACT_DIR / "catalog.parquet")
    embeddings = np.load(ARTIFACT_DIR / "embeddings.npy")
    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
    labels = metrics["head_a_genre_tagger"]["labels"]
    if len(catalog) != len(embeddings):
        raise ValueError("catalog.parquet dan embeddings.npy tidak sejajar")

    edges = knn_edges(embeddings)
    graph = build_graph(edges, len(catalog))
    logger.info("Graf kNN: %d simpul, %d sisi", graph.number_of_nodes(), graph.number_of_edges())

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    degrees = np.array([d for _, d in graph.degree()])
    pagerank = nx.pagerank(graph, weight="weight")
    communities = nx.algorithms.community.louvain_communities(graph, weight="weight", seed=SEED, resolution=1.0)
    communities = sorted(communities, key=len, reverse=True)
    modularity = nx.algorithms.community.modularity(graph, communities, weight="weight")
    community = np.empty(len(catalog), dtype=int)
    for cid, members in enumerate(communities):
        community[list(members)] = cid
    logger.info("%d komponen, %d komunitas Louvain (modularity %.4f)", len(components), len(communities), modularity)

    primary = catalog["primary_genre"].fillna("NONE").to_numpy()
    agreement = {
        "nmi_vs_kmeans": round(float(normalized_mutual_info_score(catalog["cluster_id"], community)), 4),
        "ari_vs_kmeans": round(float(adjusted_rand_score(catalog["cluster_id"], community)), 4),
        "nmi_vs_primary_genre": round(float(normalized_mutual_info_score(primary, community)), 4),
        "nmi_kmeans_vs_primary_genre": round(float(normalized_mutual_info_score(primary, catalog["cluster_id"])), 4),
    }

    pr = pd.Series(pagerank).sort_index()
    catalog = catalog.assign(pagerank=pr.to_numpy(), degree=degrees, community=community)
    hubs = catalog.nlargest(10, "pagerank")[["title", "primary_genre", "degree", "pagerank", "rating_votes"]]
    # Novel populer (banyak suara) belum tentu pusat ruang konten; korelasinya diukur, bukan diasumsikan.
    rated = catalog.dropna(subset=["rating_votes"])
    pr_vs_votes = spearmanr(rated["pagerank"], rated["rating_votes"]).statistic

    genre_sets = [set(g) if g is not None else set() for g in catalog["genre_list"]]
    homophily = genre_homophily(graph, genre_sets, labels)
    head_a = metrics["head_a_genre_tagger"]
    reference_model = head_a.get("best_model", FALLBACK_REFERENCE_MODEL)
    f1 = head_a["models"][reference_model]["f1_per_label"]
    homophily["f1"] = homophily["genre"].map(f1)
    rho_lift, p_lift = spearmanr(homophily["lift"], homophily["f1"])
    rho_share, p_share = spearmanr(homophily["neighbour_share"], homophily["f1"])
    rho_prev, p_prev = spearmanr(homophily["prevalence"], homophily["f1"])
    rho_partial = partial_spearman(homophily["f1"], homophily["neighbour_share"], homophily["prevalence"])
    logger.info("Spearman F1 vs neighbour_share = %.3f (p=%.2g)", rho_share, p_share)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "k": K_NEIGHBOURS,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "mean_edge_weight": round(float(edges["weight"].mean()), 4),
        "degree": {"min": int(degrees.min()), "median": float(np.median(degrees)), "mean": round(float(degrees.mean()), 2),
                   "p99": float(np.percentile(degrees, 99)), "max": int(degrees.max())},
        "n_components": len(components),
        "largest_component_size": len(components[0]),
        "n_communities": len(communities),
        "community_sizes_top10": [len(c) for c in communities[:10]],
        "n_communities_ge_100": sum(1 for c in communities if len(c) >= 100),
        "modularity": round(float(modularity), 4),
        "agreement": agreement,
        "pagerank_vs_votes_spearman": round(float(pr_vs_votes), 4),
        "top_pagerank": hubs.round(6).to_dict(orient="records"),
        "community_profiles": community_profiles(catalog, community),
        "homophily_reference_model": reference_model,
        "homophily": homophily.round(4).sort_values("f1", ascending=False).to_dict(orient="records"),
        "spearman_f1": {
            "neighbour_share": [round(float(rho_share), 4), float(p_share)],
            "lift": [round(float(rho_lift), 4), float(p_lift)],
            "prevalence": [round(float(rho_prev), 4), float(p_prev)],
        },
        "partial_spearman_f1_share_given_prevalence": round(rho_partial, 4),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    edges.to_parquet(EDGES_PATH, index=False)
    # Louvain peka terhadap urutan simpul; penugasan disimpan agar grafik tidak menghitung ulang partisi lain.
    catalog[["series_id", "community", "pagerank", "degree"]].to_parquet(NODES_PATH, index=False)
    logger.info("Laporan graf kemiripan ditulis ke %s", REPORT_PATH)
    if to_lake:
        publish(lake_tables(catalog, edges, community, homophily, report))
    return report


def main() -> int:
    try:
        run(to_lake="--no-lake" not in sys.argv)
    except Exception as exc:  # noqa: BLE001
        logger.error("GRAF KEMIRIPAN GAGAL: %s", exc, exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
