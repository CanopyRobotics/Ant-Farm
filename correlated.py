from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

def build_cooccurrence(orders: List) -> pd.DataFrame:
    """Build a symmetric co-occurrence matrix M where M[i,j] = #orders containing both i and j (i!=j)."""
    pairs = {}
    sku_set = set()
    for o in orders:
        skus = [int(getattr(l, 'sku_id')) for l in getattr(o, 'lines', [])]
        unique = sorted(set(skus))
        sku_set.update(unique)
        for i in range(len(unique)):
            for j in range(i+1, len(unique)):
                a, b = unique[i], unique[j]
                pairs[(a,b)] = pairs.get((a,b), 0) + 1
                pairs[(b,a)] = pairs.get((b,a), 0) + 1
    idx = sorted(sku_set)
    M = pd.DataFrame(0, index=idx, columns=idx, dtype=int)
    for (a,b), v in pairs.items():
        M.at[a,b] = v
    return M

def cluster_skus_agglomerative(M: pd.DataFrame, n_clusters: Optional[int] = None) -> List[List[int]]:
    """Cluster SKUs using Agglomerative (cosine distance). Returns list of clusters (lists of sku_ids)."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_distances
    if M.empty:
        return []
    # Use normalized rows (affinity vectors)
    X = M.values.astype(float)
    # Avoid zero rows producing NaNs
    row_sums = X.sum(axis=1)
    nz = row_sums > 0
    if not nz.any():
        # fallback: one cluster per SKU
        return [[int(sku)] for sku in M.index.to_list()]
    Xn = X.copy()
    Xn[nz] = X[nz] / row_sums[nz][:, None]
    D = cosine_distances(Xn)
    # Heuristic cluster count if not provided: sqrt of n
    k = n_clusters or max(1, int(np.sqrt(X.shape[0])))
    model = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average')
    labels = model.fit_predict(D)
    clusters: Dict[int,List[int]] = {}
    skus = M.index.to_list()
    for sku, lab in zip(skus, labels):
        clusters.setdefault(int(lab), []).append(int(sku))
    return list(clusters.values())

def sample_correlated_orders(
    clusters: List[List[int]],
    sku_popularity: Optional[Dict[int, float]] = None,
    num_orders: int = 100,
    mean_lines: float = 4.0,
    cross_cluster_prob: float = 0.2,
) -> List:
    """Sample orders by drawing cluster first, then items within cluster, with occasional cross-cluster picks."""
    import random
    from models import Order, OrderLine
    rng = random.Random(42)
    # cluster weights proportional to sum popularity or size
    def group_score(g):
        if sku_popularity:
            return sum(sku_popularity.get(s, 1.0) for s in g)
        return float(len(g))
    scores = [group_score(g) for g in clusters]
    total = sum(scores) or 1.0
    probs = [s/total for s in scores]
    # per-item weights inside cluster
    item_wts = []
    for g in clusters:
        if sku_popularity:
            ws = [sku_popularity.get(s, 1.0) for s in g]
        else:
            ws = [1.0 for _ in g]
        item_wts.append(ws)

    orders: List[Order] = []
    for oid in range(1, num_orders+1):
        # Poisson-like lines count using exponential
        lines = max(1, int(rng.expovariate(1.0/mean_lines)))
        # pick a cluster
        ci = rng.choices(range(len(clusters)), weights=probs, k=1)[0]
        cluster = clusters[ci]
        wts = item_wts[ci]
        chosen = rng.choices(cluster, weights=wts, k=lines)
        # occasional cross-cluster items
        if rng.random() < cross_cluster_prob and len(clusters) > 1:
            cj = ci
            attempts = 0
            while cj == ci and attempts < 5:
                cj = rng.randrange(len(clusters)); attempts += 1
            other = clusters[cj]
            owts = item_wts[cj]
            extra = rng.choices(other, weights=owts, k=max(1, lines//4))
            chosen.extend(extra)
        # create order
        olines = [OrderLine(sku_id=int(s)) for s in chosen]
        orders.append(Order(id=oid, lines=olines))
    return orders

def tote_pick_counts(orders: List, sku_to_tote: Dict[int, str]) -> Dict[str, int]:
    """Aggregate pick counts per tote_id given orders and sku->tote mapping."""
    counts: Dict[str,int] = {}
    for o in orders:
        for ln in getattr(o, 'lines', []):
            tid = sku_to_tote.get(int(ln.sku_id))
            if tid is None:
                continue
            counts[tid] = counts.get(tid, 0) + 1
    return counts
