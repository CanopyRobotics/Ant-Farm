# main.py

from models import WarehouseCfg, Order
from storage import gen_storage_locations
from slotting import (
    RoundRobinSlotting,
    PopularityABCSlotting,
    RandomSlotting,
    AffinitySlotting
)
from batching import RoundRobinBatching
from routing import SideGroupedRouting
from simulation import SimulationEngine
from visualization import plot_warehouse_map
from kpis import compute_kpis
from affinity import compute_affinity_matrix, group_skus_by_affinity

import random
import numpy as np

def gen_skus(num_aisles, sections_per_side):
    num_locations = num_aisles * 2 * sections_per_side * 6 * 3
    return list(range(1, num_locations + 1))

def gen_orders(num_orders, sku_ids, mean_lines, rng, sku_popularity=None):
    from models import OrderLine
    orders = []
    weights = None
    if sku_popularity:
        weights = [sku_popularity[sku] for sku in sku_ids]
    for oid in range(1, num_orders + 1):
        lines = max(1, int(rng.expovariate(1.0 / mean_lines)))
        if weights:
            chosen = rng.choices(sku_ids, weights=weights, k=lines)
        else:
            chosen = [rng.choice(sku_ids) for _ in range(lines)]
        orders.append(Order(id=oid, lines=[OrderLine(sku_id=s) for s in chosen]))
    return orders

def generate_pareto_popularity(sku_ids, alpha=1.16):
    """
    Generates a Pareto-distributed popularity for SKUs.
    alpha ≈ 1.16 gives a strong 80/20 effect.
    Returns a dict: {sku_id: popularity}
    """
    n = len(sku_ids)
    raw = np.random.pareto(alpha, n) + 1  # +1 to avoid zeros
    popularity = raw / raw.sum()
    return dict(zip(sku_ids, popularity))

def main():
    # --- Warehouse and Data Generation ---
    wh = WarehouseCfg(num_aisles=10, aisle_length_m=20.0, aisle_width_m=3.0, speed_mps=0.75, pick_time_s=20.0)
    sections_per_side = 10
    num_pickers = 3
    num_orders = 25 * num_pickers
    mean_lines = 4
    rng = random.Random(42)
    sku_ids = gen_skus(wh.num_aisles, sections_per_side)
    sku_popularity = generate_pareto_popularity(sku_ids)  # <-- Add this before gen_orders!
    orders = gen_orders(num_orders, sku_ids, mean_lines, rng, sku_popularity)
    locations = gen_storage_locations(wh.num_aisles, sections_per_side)

    # --- Affinity Analysis (for AffinitySlotting) ---
    affinity = compute_affinity_matrix(orders)
    sku_groups = group_skus_by_affinity(affinity, sku_ids, group_size=18)  # 18 = totes per section/side

    # --- Choose Slotting Policy ---
    # Uncomment ONE of the following slotting policies:
    # slotting_policy = AffinitySlotting(sku_groups)           # Affinity/family-based slotting
    slotting_policy = PopularityABCSlotting(sku_popularity) # Popularity/ABC slotting
    # slotting_policy = RandomSlotting()                      # Random slotting
    # slotting_policy = RoundRobinSlotting()                  # Round-robin slotting

    # --- Choose Batching Policy ---
    batching_policy = RoundRobinBatching()

    # --- Choose Routing Policy ---
    routing_policy = SideGroupedRouting()

    # --- Simulation ---
    sim = SimulationEngine(wh, slotting_policy, batching_policy, routing_policy)
    orders_by_picker = batching_policy.batch(orders, num_pickers)  # <-- Move this up!
    picker_paths = sim.run(sku_ids, orders, locations, sections_per_side, num_pickers=num_pickers)
    kpis = compute_kpis(picker_paths, orders_by_picker, wh)

    # --- ABC Classification for Visualization ---
    sorted_skus = sorted(sku_popularity, key=sku_popularity.get, reverse=True)
    n = len(sorted_skus)
    a_skus = set(sorted_skus[:int(0.2 * n)])
    b_skus = set(sorted_skus[int(0.2 * n):int(0.5 * n)])
    c_skus = set(sorted_skus[int(0.5 * n):])

    # --- Visualization ---
    sku_to_location = slotting_policy.assign(sku_ids, locations)

    plot_warehouse_map(
        wh.num_aisles, sections_per_side, wh,
        orders_by_picker, sku_to_location, picker_paths, kpis,
        a_skus, b_skus, c_skus
    )
    # # Pritning ABC class counts per picker for PopularityABCSlotting
    # for picker_idx in range(3):
    #     a_count = 0
    #     b_count = 0
    #     c_count = 0
    #     for order in orders_by_picker[picker_idx]:
    #         for line in order.lines:
    #             sku = line.sku_id
    #             if sku in a_skus:
    #                 a_count += 1
    #             elif sku in b_skus:
    #                 b_count += 1
    #             elif sku in c_skus:
    #                 c_count += 1
    #     print(f"\nPicker {picker_idx+1} ABC class counts:")
    #     print(f"A class SKUs: {a_count}")
    #     print(f"B class SKUs: {b_count}")
    #     print(f"C class SKUs: {c_count}")


if __name__ == "__main__":
    main()