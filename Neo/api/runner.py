from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import json

# Reuse existing modules
from affinity import compute_affinity_matrix as af_compute_affinity_matrix, group_skus_by_affinity as af_group_skus_by_affinity
from models import Order
from slotting import AffinitySlotting, PopularityABCSlotting, RandomSlotting, RoundRobinSlotting
from correlated import build_cooccurrence, cluster_skus_agglomerative, sample_correlated_orders, tote_pick_counts


def _enforce_one_sku_per_tote(assign_map, layout_df, sku_ids):
    loc_cols = ["side_id", "section", "tote_id"]
    free = list(layout_df["tote_id"])  # layout order
    taken = set()
    resolved = {}

    def take_tote(tid):
        if tid in taken:
            return None
        taken.add(tid)
        free.remove(tid)
        return tid

    for sku in sorted(sku_ids):
        loc_obj = assign_map.get(sku)
        if loc_obj is None:
            resolved[sku] = None
            continue
        d = loc_obj.__dict__ if hasattr(loc_obj, "__dict__") else loc_obj
        tid = d["tote_id"]
        if tid in free:
            take_tote(tid)
            resolved[sku] = d
        else:
            resolved[sku] = None

    idx_by_tote = layout_df.set_index("tote_id")["side_id"].to_dict(), layout_df.set_index("tote_id")["section"].to_dict()
    side_by_tote, sec_by_tote = idx_by_tote
    for sku in sorted(sku_ids):
        if resolved[sku] is None:
            if not free:
                raise RuntimeError("Not enough totes to enforce 1 SKU per tote.")
            tid = free[0]
            take_tote(tid)
            resolved[sku] = {"side_id": side_by_tote[tid], "section": sec_by_tote[tid], "tote_id": tid}

    return resolved


def run_slotting(layout_df: pd.DataFrame, sku_df: pd.DataFrame, sales_df: pd.DataFrame | None,
                 policy: str = "affinity", simulate_correlated: bool = False, optimization: str = "max") -> Dict[str, Any]:
    # Current map
    current_map = sku_df.merge(layout_df[["tote_id", "side_id", "section"]], on="tote_id", how="left")
    # Locations list
    loc_cols = ["side_id", "section", "tote_id"]
    from dataclasses import make_dataclass
    StorageLocation = make_dataclass("StorageLocation", [(c, str) for c in loc_cols])
    locations = [StorageLocation(**row) for row in layout_df[loc_cols].to_dict("records")]

    sku_ids = pd.Series(current_map["sku_id"]).dropna().astype(int).unique().tolist()
    sku_ids.sort()
    if len(sku_ids) > len(locations):
        if sales_df is not None and "sku_id" in sales_df.columns and "quantity" in sales_df.columns:
            sales_rank = sales_df.groupby("sku_id")["quantity"].sum().sort_values(ascending=False)
            top_skus = [int(s) for s in sales_rank.index.tolist() if int(s) in set(sku_ids)][:len(locations)]
            if len(top_skus) < len(locations):
                remaining = [s for s in sku_ids if s not in set(top_skus)]
                top_skus.extend(remaining[:len(locations)-len(top_skus)])
            sku_ids = top_skus
        else:
            sku_ids = sku_ids[:len(locations)]

    if policy == "affinity":
        if sales_df is None or "order_id" not in (sales_df.columns if sales_df is not None else []):
            pop = sales_df.groupby("sku_id")["quantity"].sum().to_dict() if sales_df is not None else {}
            policy_obj = PopularityABCSlotting(pop)
            assign = policy_obj.assign(sku_ids, locations)
            groups_used = []
        else:
            order_groups = sales_df.groupby("order_id")["sku_id"].apply(list)
            orders = [Order(id=int(oid), lines=[type("OL", (), {"sku_id": int(s)})() for s in skus]) for oid, skus in order_groups.items()]
            affinity = af_compute_affinity_matrix(orders)
            # group size inferred from layout (mode totes per section)
            sec_counts = layout_df.groupby(["side_id","section"])['tote_id'].nunique().tolist()
            group_size = max(1, pd.Series(sec_counts).mode().iloc[0]) if len(sec_counts) else 18
            groups = af_group_skus_by_affinity(affinity, sku_ids, group_size=group_size)
            sku_pop_map = {}
            if sales_df is not None and "quantity" in sales_df.columns:
                pop_series = sales_df.groupby("sku_id")["quantity"].sum()
                sku_pop_map = {int(s): float(pop_series.get(int(s), 0.0)) for s in sku_ids}
            policy_obj = AffinitySlotting(groups, sku_popularity=sku_pop_map)
            assign = policy_obj.assign(sku_ids, locations)
            groups_used = [list(g) for g in groups]
    elif policy == "abc":
        pop = sales_df.groupby("sku_id")["quantity"].sum().to_dict() if sales_df is not None else {}
        policy_obj = PopularityABCSlotting(pop)
        assign = policy_obj.assign(sku_ids, locations)
        groups_used = []
    elif policy == "rr":
        assign = RoundRobinSlotting().assign(sku_ids, locations)
        groups_used = []
    else:
        assign = RandomSlotting().assign(sku_ids, locations)
        groups_used = []

    assign_1to1 = _enforce_one_sku_per_tote(assign, layout_df, sku_ids)
    proposed_map = pd.DataFrame([{ "sku_id": sku, **assign_1to1[sku] } for sku in sku_ids])

    tote_heat_override = None
    if simulate_correlated and sales_df is not None and "order_id" in sales_df.columns:
        # use same groups for simulation when available
        if not groups_used:
            order_groups = sales_df.groupby("order_id")["sku_id"].apply(list)
            orders_raw = [Order(id=int(oid), lines=[type("OL", (), {"sku_id": int(s)})() for s in skus]) for oid, skus in order_groups.items()]
            M = build_cooccurrence(orders_raw)
            groups_used = cluster_skus_agglomerative(M)
        sku_to_tote = {int(row["sku_id"]): row["tote_id"] for _, row in proposed_map.iterrows()}
        # popularity weights
        sku_pop = sales_df.groupby("sku_id")["quantity"].sum().to_dict()
        num_orders_obs = int(sales_df["order_id"].nunique())
        mean_lines_obs = float(sales_df.groupby("order_id")["sku_id"].size().mean()) if num_orders_obs > 0 else 4.0
        syn_orders = sample_correlated_orders(groups_used, sku_popularity=sku_pop, num_orders=num_orders_obs, mean_lines=mean_lines_obs, cross_cluster_prob=0.15)
        tote_heat_override = tote_pick_counts(syn_orders, sku_to_tote)
        # scale to match observed volume
        total_current = float(sales_df["quantity"].sum()) if "quantity" in sales_df.columns else float(len(sales_df))
        total_sim = float(sum(tote_heat_override.values())) or 1.0
        scale = total_current / total_sim
        tote_heat_override = {k: v * scale for k, v in tote_heat_override.items()}

    return {
        "current_map": current_map.to_dict("records"),
        "proposed_map": proposed_map.to_dict("records"),
        "tote_heat_override": tote_heat_override or {},
    }


def build_move_plan_csv(proposed_map_json: str, current_map_json: str, sales_df: pd.DataFrame | None, layout_df: pd.DataFrame | None, top_n: int = 1000) -> str:
    cur = pd.DataFrame(json.loads(current_map_json))
    prop = pd.DataFrame(json.loads(proposed_map_json))
    if layout_df is not None:
        for df in (cur, prop):
            if 'side_id' not in df.columns or 'section' not in df.columns:
                df.drop(columns=[c for c in df.columns if c not in ('sku_id','tote_id')], inplace=True)
                df = df.merge(layout_df[["tote_id","side_id","section"]], on="tote_id", how="left")
    if sales_df is not None and 'sku_id' in sales_df.columns and 'quantity' in sales_df.columns:
        pop = sales_df.groupby('sku_id')['quantity'].sum()
    else:
        pop = pd.Series(1, index=pd.Index(cur['sku_id'].unique(), name='sku_id'))

    merged = cur[['sku_id','tote_id','side_id','section']].rename(columns={'tote_id':'from_tote','side_id':'from_side','section':'from_section'}).merge(
        prop[['sku_id','tote_id','side_id','section']].rename(columns={'tote_id':'to_tote','side_id':'to_side','section':'to_section'}), on='sku_id', how='inner'
    )
    def idx(letters: str) -> int:
        s = str(letters).split('-')[-1] if isinstance(letters, str) and '-' in str(letters) else str(letters)
        if not s:
            return 0
        s = ''.join([c for c in s if c.isalpha()])
        if len(s) == 1:
            return ord(s[0]) - ord('A')
        return (ord(s[0]) - ord('A')) * 26 + (ord(s[1]) - ord('A'))
    merged['from_idx'] = merged['from_section'].apply(idx)
    merged['to_idx'] = merged['to_section'].apply(idx)
    merged['delta_sections'] = (merged['from_idx'] - merged['to_idx']).clip(lower=0)
    merged['popularity'] = merged['sku_id'].map(pop).fillna(0).astype(float)
    merged['impact_score'] = merged['popularity'] * merged['delta_sections']
    plan = merged[merged['from_tote'] != merged['to_tote']].copy()
    plan.sort_values(['impact_score','popularity','delta_sections'], ascending=[False, False, False], inplace=True)
    out = plan.head(int(top_n))[[
        'sku_id','from_tote','from_side','from_section','to_tote','to_side','to_section','popularity','delta_sections','impact_score'
    ]]
    return out.to_csv(index=False)


def build_mapping_csv(proposed_map_json: str, layout_df: pd.DataFrame | None) -> str:
    prop = pd.DataFrame(json.loads(proposed_map_json))
    if layout_df is not None and ('side_id' not in prop.columns or 'section' not in prop.columns):
        prop = prop.merge(layout_df[["tote_id","side_id","section"]], on="tote_id", how="left")
    out = prop[['sku_id','tote_id','side_id','section']].sort_values(['side_id','section','tote_id','sku_id'])
    return out.to_csv(index=False)
