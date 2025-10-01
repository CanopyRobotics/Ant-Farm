"""
Generate two CSVs to test Neo with Ant-Farm-like randomness:
1) sales.csv: columns [order_id, order_timestamp, sku_id, quantity, tote_id, side_id, section]
2) sku_locations.csv: columns [sku_id, tote_id, quantity]

It uses the same naming convention for side_id, section, and tote_id as Ant-Farm (e.g., A003, A003-AB, A003-AB-032).
The generator creates a warehouse layout, assigns SKUs to totes, samples a popularity distribution for SKUs,
then generates orders with random lines whose SKUs inherit the tote/location they were stored in at time of sale.

Usage:
  python tools/generate_neo_csvs.py --num-aisles 8 --sections-per-side 10 --num-skus 1500 --num-orders 3000 --seed 42 --out-dir run_data
"""
from __future__ import annotations
import argparse
import os
import random
from datetime import datetime, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd

# Reuse canonical storage naming from project
try:
    from storage import gen_storage_locations
except Exception:
    # Fallback for direct execution when working dir differs
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from storage import gen_storage_locations


def gen_skus(num_skus: int) -> List[int]:
    return list(range(100000, 100000 + num_skus))


def sample_popularity(num_skus: int, alpha: float = 1.07) -> np.ndarray:
    """Zipf-like popularity weights (lower alpha -> heavier head)."""
    ranks = np.arange(1, num_skus + 1)
    weights = 1 / np.power(ranks, alpha)
    return weights / weights.sum()


def assign_skus_to_locations_random_one_to_one(locations: List[dict]) -> Dict[int, dict]:
    """Assign one unique SKU per tote in random order (100% occupancy)."""
    import random as _random
    locs = locations.copy()
    _random.shuffle(locs)
    sku_ids = gen_skus(len(locs))
    return {sku: loc for sku, loc in zip(sku_ids, locs)}


def make_layout_df(num_aisles: int, sections_per_side: int) -> pd.DataFrame:
    # Build all totes just like storage.gen_storage_locations
    locs = gen_storage_locations(num_aisles, sections_per_side)
    return pd.DataFrame([{"side_id": l.side_id, "section": l.section, "tote_id": l.tote_id} for l in locs])


def build_sku_locations_df(sku_to_loc: Dict[int, dict]) -> pd.DataFrame:
    # quantity in tote: 1..30 (cap to simulate tote capacity)
    qtys = np.random.poisson(lam=8, size=len(sku_to_loc)).clip(min=1, max=30)
    rows = []
    for (sku, loc), q in zip(sku_to_loc.items(), qtys):
        rows.append({"sku_id": sku, "tote_id": loc["tote_id"], "quantity": int(q)})
    return pd.DataFrame(rows)


def generate_orders(sku_ids: List[int], sku_to_loc: Dict[int, dict], num_orders: int, start_date: datetime) -> pd.DataFrame:
    # Popularity driven SKU sampling
    probs = sample_popularity(len(sku_ids))
    sku_arr = np.array(sku_ids)

    # Order sizes: 1-6 lines; mildly geometric
    line_counts = np.random.choice([1,2,3,4,5,6], size=num_orders, p=[0.25,0.25,0.2,0.15,0.1,0.05])

    order_rows = []
    order_id_base = 500000
    cursor = start_date
    for idx, lines in enumerate(line_counts):
        order_id = order_id_base + idx
        # random time increments within a 30 day window
        cursor += timedelta(minutes=int(np.random.exponential(scale=30)))
        for _ in range(lines):
            sku = int(np.random.choice(sku_arr, p=probs))
            qty = int(max(1, np.random.poisson(lam=2)))
            loc = sku_to_loc[sku]
            order_rows.append({
                "order_id": order_id,
                "order_timestamp": cursor.strftime("%Y-%m-%d %H:%M:%S"),
                "sku_id": sku,
                "quantity": qty,
                "tote_id": loc["tote_id"],
                "side_id": loc["side_id"],
                "section": loc["section"],
            })
    return pd.DataFrame(order_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-aisles", type=int, default=10)
    ap.add_argument("--sections-per-side", type=int, default=10)
    ap.add_argument("--num-skus", type=int, default=3000)
    ap.add_argument("--num-orders", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="run_data")
    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    # Layout and SKU placement
    layout_df = make_layout_df(args.num_aisles, args.sections_per_side)

    # Prepare a compact list of unique locations
    locations = layout_df.to_dict("records")
    num_totes = len(locations)

    # Generate random one-to-one assignment (100% occupancy baseline, 1 SKU per tote)
    sku_to_loc = assign_skus_to_locations_random_one_to_one(locations)
    sku_ids = list(sku_to_loc.keys())

    # Build sku_locations.csv (quantity in totes)
    sku_locations_df = build_sku_locations_df(sku_to_loc)

    # Build sales.csv with Ant-Farm-like randomness
    start_date = datetime.now() - timedelta(days=30)
    sales_df = generate_orders(sku_ids, sku_to_loc, args.num_orders, start_date)

    # Output
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    sales_path = os.path.join(out_dir, "sales.csv")
    skus_path = os.path.join(out_dir, "sku_locations.csv")
    layout_path = os.path.join(out_dir, "layout.csv")

    sales_df.to_csv(sales_path, index=False)
    sku_locations_df.to_csv(skus_path, index=False)
    layout_df.to_csv(layout_path, index=False)

    print(f"Wrote: {sales_path}\n       : {skus_path}\n       : {layout_path}\n       Info: Generated {num_totes} SKUs for {num_totes} totes (100% occupancy, 1 SKU/tote, qty<=30)")


if __name__ == "__main__":
    main()
