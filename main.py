# main.py

from models import WarehouseCfg, Order
from storage import gen_storage_locations
from slotting import (
    RoundRobinSlotting,
    PopularityABCSlotting,
    RandomSlotting,
    AffinitySlotting
)
from batching import GreedyProximityBatching, RoundRobinBatching, BatchingPolicyAdapter, SeedSavingsBatching, RandomBatching
from routing import SideGroupedRouting, SShapeRouting, LargestGapRouting, HybridCombinedRouting
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
    raw = np.random.pareto(alpha, n) + 1
    popularity = raw / raw.sum()
    return dict(zip(sku_ids, popularity))

def main():
    # --- Warehouse and Data Generation ---
    wh = WarehouseCfg(num_aisles=10, aisle_length_m=20.0, aisle_width_m=3.0, speed_mps=0.75, pick_time_s=20.0)
    sections_per_side = 10
    num_pickers = 1
    num_orders = 25 * num_pickers
    mean_lines = 4
    rng = random.Random(42)
    sku_ids = gen_skus(wh.num_aisles, sections_per_side)
    sku_popularity = generate_pareto_popularity(sku_ids)
    orders = gen_orders(num_orders, sku_ids, mean_lines, rng, sku_popularity)
    locations = gen_storage_locations(wh.num_aisles, sections_per_side)

    # --- Affinity Analysis (for AffinitySlotting) ---
    affinity = compute_affinity_matrix(orders)
    sku_groups = group_skus_by_affinity(affinity, sku_ids, group_size=18)

    # --- Choose Slotting Policy ---
    # slotting_policy = AffinitySlotting(sku_groups)
    # slotting_policy = PopularityABCSlotting(sku_popularity)
    # slotting_policy = RandomSlotting()
    slotting_policy = RoundRobinSlotting()

    sku_to_location = slotting_policy.assign(sku_ids, locations)

    # --- Choose Batching Policy ---
    # batching_policy = BatchingPolicyAdapter(GreedyProximityBatching())
    # batching_policy = BatchingPolicyAdapter(SeedSavingsBatching())
    # batching_policy = BatchingPolicyAdapter(RoundRobinBatching())
    batching_policy = BatchingPolicyAdapter(RandomBatching())


    orders_by_picker = batching_policy.batch(
        orders, num_pickers, sku_to_location, wh, sections_per_side, max_orders_per_picker=25
    )

    # --- Choose Routing Policy ---
    routing_policy = SideGroupedRouting()
    # routing_policy = SShapeRouting()
    # routing_policy = LargestGapRouting()
    # routing_policy = HybridCombinedRouting()

    # --- Simulation ---
    sim = SimulationEngine(wh, slotting_policy, batching_policy, routing_policy)
    picker_paths = sim.run(sku_ids, orders, locations, sections_per_side, num_pickers=num_pickers)
    kpis = compute_kpis(picker_paths, orders_by_picker, wh)

    # --- ABC Classification for Visualization ---
    sorted_skus = sorted(sku_popularity, key=sku_popularity.get, reverse=True)
    n = len(sorted_skus)
    a_skus = set(sorted_skus[:int(0.2 * n)])
    b_skus = set(sorted_skus[int(0.2 * n):int(0.5 * n)])
    c_skus = set(sorted_skus[int(0.5 * n):])

    # --- Visualization ---
    plot_warehouse_map(
        wh.num_aisles, sections_per_side, wh,
        orders_by_picker, sku_to_location, picker_paths, kpis,
        a_skus, b_skus, c_skus
    )

if __name__ == "__main__":
    main()