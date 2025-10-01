import pandas as pd
from typing import Tuple

# Expected schemas (flexible but documented)
# layout.csv: aisle:int, side_id:str, section:str, tote_id:str
# sku_locations.csv: sku_id:int, tote_id:str
# sales.csv: order_id:str|int, sku_id:int, quantity:int, order_timestamp:str

def read_layout(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"aisle", "side_id", "section", "tote_id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"layout missing columns: {missing}")
    return df


def read_sku_locations(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"sku_id", "tote_id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"sku_locations missing columns: {missing}")
    return df


def read_sales(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"order_id", "sku_id", "quantity"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"sales missing columns: {missing}")
    return df


def build_affinity_from_sales(sales: pd.DataFrame) -> dict:
    # Simple co-occurrence affinity by order_id
    groups = sales.groupby("order_id")["sku_id"].apply(set)
    from collections import defaultdict
    aff = defaultdict(int)
    for skus in groups:
        skus = list(skus)
        for i in range(len(skus)):
            for j in range(i + 1, len(skus)):
                a, b = sorted((skus[i], skus[j]))
                aff[(a, b)] += 1
    return dict(aff)


def group_skus_by_affinity(sales: pd.DataFrame, group_size: int = 18) -> list[list[int]]:
    """Greedy grouping mirroring Ant-Farm semantics.
    Key differences from the previous version:
    - No global popularity sorting; start groups in arbitrary set order to avoid positional bias.
    - For each group, iteratively add the SKU with highest co-occurrence with the current group.
    This matches affinity.group_skus_by_affinity used in Ant-Farm and prevents systemic left/right bias
    when locations are consumed in reading order.
    """
    aff_pairs = build_affinity_from_sales(sales)  # dict[(a,b)] -> count
    skus = list(sales["sku_id"].unique())
    unassigned = set(skus)
    groups: list[list[int]] = []

    def pair_aff(a: int, b: int) -> int:
        if a == b:
            return 0
        x, y = (a, b) if a < b else (b, a)
        return aff_pairs.get((x, y), 0)

    while unassigned:
        # Start a new group with an arbitrary SKU (set pop -> pseudo-random order)
        sku = unassigned.pop()
        group = [sku]
        # Add best partners by total affinity to the current group
        for _ in range(group_size - 1):
            if not unassigned:
                break
            # choose the s with max sum(pair_aff(s, g) for g in group)
            best = max(unassigned, key=lambda s: sum(pair_aff(s, g) for g in group))
            group.append(best)
            unassigned.remove(best)
        groups.append(group)

    return groups
