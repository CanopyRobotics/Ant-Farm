# main.py

from models import WarehouseCfg, Order
from storage import gen_storage_locations, assign_skus_to_locations
from slotting import RoundRobinSlotting
from slotting import PopularityABCSlotting
from slotting import RandomSlotting
from batching import RoundRobinBatching
from routing import SideGroupedRouting
from simulation import SimulationEngine
from visualization import plot_warehouse_map
from kpis import compute_kpis

import random
import numpy as np

def gen_skus(num_aisles, sections_per_side):
    num_locations = num_aisles * 2 * sections_per_side * 6 * 3
    return list(range(1, num_locations + 1))

def gen_orders(num_orders, sku_ids, mean_lines, rng):
    from models import OrderLine
    orders = []
    for oid in range(1, num_orders + 1):
        lines = max(1, int(rng.expovariate(1.0 / mean_lines)))
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
    wh = WarehouseCfg(num_aisles=10, aisle_length_m=20.0, aisle_width_m=3.0, speed_mps=0.75, pick_time_s=20.0)
    sections_per_side = 10
    num_pickers = 1
    num_orders = 25
    mean_lines = 4
    rng = random.Random(42)
    sku_ids = gen_skus(wh.num_aisles, sections_per_side)
    locations = gen_storage_locations(wh.num_aisles, sections_per_side)
    orders = gen_orders(num_orders, sku_ids, mean_lines, rng)
    sku_popularity = generate_pareto_popularity(sku_ids)
    slotting_policy = PopularityABCSlotting(sku_popularity)
    # slotting_policy = RandomSlotting()
    batching_policy = RoundRobinBatching()
    routing_policy = SideGroupedRouting()
    sim = SimulationEngine(wh, slotting_policy, batching_policy, routing_policy)
    picker_path = sim.run(sku_ids, orders, locations, sections_per_side, num_pickers=num_pickers)
    kpis = compute_kpis(picker_path, orders, wh)
    # Sort SKUs by popularity
    sorted_skus = sorted(sku_popularity, key=sku_popularity.get, reverse=True)
    n = len(sorted_skus)
    a_skus = set(sorted_skus[:int(0.2 * n)])
    b_skus = set(sorted_skus[int(0.2 * n):int(0.5 * n)])
    c_skus = set(sorted_skus[int(0.5 * n):])
    # For visualization, you may need to re-batch and re-slot for the plot
    orders_by_picker = batching_policy.batch(orders, num_pickers)
    sku_to_location = slotting_policy.assign(sku_ids, locations)
    plot_warehouse_map(wh.num_aisles, sections_per_side, wh, orders_by_picker, sku_to_location, picker_path, kpis, a_skus, b_skus, c_skus)

if __name__ == "__main__":
    main()