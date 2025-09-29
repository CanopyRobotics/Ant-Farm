import csv
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models import WarehouseCfg
from storage import gen_storage_locations
from slotting import AffinitySlotting, PopularityABCSlotting, RandomSlotting, RoundRobinSlotting
from batching import GreedyProximityBatching, SeedSavingsBatching, RoundRobinBatching, RandomBatching, BatchingPolicyAdapter
from routing import SideGroupedRouting, SShapeRouting, LargestGapRouting, HybridCombinedRouting
from simulation import SimulationEngine
from kpis import compute_kpis
from affinity import compute_affinity_matrix, group_skus_by_affinity

def gen_skus(num_aisles, sections_per_side):
    num_locations = num_aisles * 2 * sections_per_side * 6 * 3
    return list(range(1, num_locations + 1))

def gen_orders(num_orders, sku_ids, mean_lines, rng, sku_popularity=None):
    from models import OrderLine, Order
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
    n = len(sku_ids)
    raw = np.random.pareto(alpha, n) + 1
    popularity = raw / raw.sum()
    return dict(zip(sku_ids, popularity))

def single_run(args):
    slot_name, slot_policy_fn, batch_name, batch_policy_fn, route_name, route_policy_fn, iteration = args
    wh = WarehouseCfg(num_aisles=10, aisle_length_m=20.0, aisle_width_m=3.0, speed_mps=0.75, pick_time_s=20.0)
    sections_per_side = 10
    num_pickers = 4
    num_orders = 25 * num_pickers
    mean_lines = 4
    rng = random.Random(iteration)
    sku_ids = gen_skus(wh.num_aisles, sections_per_side)
    sku_popularity = generate_pareto_popularity(sku_ids)
    orders = gen_orders(num_orders, sku_ids, mean_lines, rng, sku_popularity)
    locations = gen_storage_locations(wh.num_aisles, sections_per_side)

    if slot_name == "AffinitySlotting":
        slotting_policy = slot_policy_fn(sku_ids, locations, orders)
    elif slot_name == "PopularityABCSlotting":
        slotting_policy = slot_policy_fn(sku_ids, locations, orders, sku_popularity)
    else:
        slotting_policy = slot_policy_fn(sku_ids, locations, orders)

    sku_to_location = slotting_policy.assign(sku_ids, locations)
    batching_policy = batch_policy_fn()
    orders_by_picker = batching_policy.batch(
        orders, num_pickers, sku_to_location, wh, sections_per_side, max_orders_per_picker=25
    )
    routing_policy = route_policy_fn()
    sim = SimulationEngine(wh, slotting_policy, batching_policy, routing_policy)
    picker_paths = sim.run(sku_ids, orders, locations, sections_per_side, num_pickers=num_pickers)
    kpis = compute_kpis(picker_paths, orders_by_picker, wh)
    op_kpis = kpis["Operation"]
    row = {
        "Iteration": iteration,
        "Slotting": slot_name,
        "Batching": batch_name,
        "Routing": route_name,
        "Total Distance Walked (m)": op_kpis["Total Distance Walked (m)"],
        "Total Time (min)": op_kpis["Total Time (min)"],
        "Average Picker Distance (m)": op_kpis["Average Picker Distance (m)"],
        "Average Picker Time (min)": op_kpis["Average Picker Time (min)"],
        "Max Picker Distance (m)": op_kpis["Max Picker Distance (m)"],
        "Min Picker Distance (m)": op_kpis["Min Picker Distance (m)"],
        "Average Orders per Picker": op_kpis["Average Orders per Picker"],
        "Average Lines per Order": op_kpis["Average Lines per Order"],
        "Average Batch Size": op_kpis["Average Batch Size"],
        "SKU Overlap": op_kpis["SKU Overlap"],
        "Average Aisles Visited": op_kpis["Average Aisles Visited"],
        "Average Penetration Depth": op_kpis["Average Penetration Depth"],
        "Picks by Class (%)": op_kpis["Picks by Class (%)"],
        "Average Distance to A-class SKU": op_kpis["Average Distance to A-class SKU"],
        "Orders Variance": op_kpis["Picker Workload Balance"]["Orders Variance"],
        "Lines Variance": op_kpis["Picker Workload Balance"]["Lines Variance"],
        "Distance Variance": op_kpis["Picker Workload Balance"]["Distance Variance"]
    }
    return row

def run_experiments():
    wh = WarehouseCfg(num_aisles=10, aisle_length_m=20.0, aisle_width_m=3.0, speed_mps=0.75, pick_time_s=20.0)
    sections_per_side = 10
    num_pickers = 3
    num_orders = 25 * num_pickers
    mean_lines = 4

    # --- Add timestamp to filename ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "runallcombos_results"
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = os.path.join(results_dir, f"experiment_results_{timestamp}.csv")

    slotting_policies = [
        ("AffinitySlotting", make_affinity_slotting),
        ("PopularityABCSlotting", make_popularity_abc_slotting),
        ("RandomSlotting", make_random_slotting),
        ("RoundRobinSlotting", make_round_robin_slotting)
    ]

    batching_policies = [
        ("GreedyProximityBatching", make_greedy_batching),
        ("SeedSavingsBatching", make_seed_savings_batching),
        ("RoundRobinBatching", make_round_robin_batching),
        ("RandomBatching", make_random_batching)
    ]

    routing_policies = [
        ("SideGroupedRouting", make_side_grouped_routing),
        ("SShapeRouting", make_s_shape_routing),
        ("LargestGapRouting", make_largest_gap_routing),
        ("HybridCombinedRouting", make_hybrid_combined_routing)
    ]

    all_args = [
        (slot_name, slot_policy_fn, batch_name, batch_policy_fn, route_name, route_policy_fn, iteration)
        for slot_name, slot_policy_fn in slotting_policies
        for batch_name, batch_policy_fn in batching_policies
        for route_name, route_policy_fn in routing_policies
        for iteration in range(1, 101)
    ]

    with open(csv_filename, "w", newline='') as csvfile:
        fieldnames = [
            "Iteration", "Slotting", "Batching", "Routing",
            "Total Distance Walked (m)", "Total Time (min)", "Average Picker Distance (m)",
            "Average Picker Time (min)", "Max Picker Distance (m)", "Min Picker Distance (m)",
            "Average Orders per Picker", "Average Lines per Order", "Average Batch Size",
            "SKU Overlap", "Average Aisles Visited", "Average Penetration Depth",
            "Picks by Class (%)", "Average Distance to A-class SKU",
            "Orders Variance", "Lines Variance", "Distance Variance"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with ProcessPoolExecutor() as executor:
            for row in tqdm(executor.map(single_run, all_args), total=len(all_args), desc="Experiment Progress"):
                writer.writerow(row)

    # --- Data Analysis and Visualization ---
    df = pd.read_csv(csv_filename)
    # Convert numeric columns if needed
    df['Total Distance Walked (m)'] = pd.to_numeric(df['Total Distance Walked (m)'], errors='coerce')
    df['Average Picker Time (min)'] = pd.to_numeric(df['Average Picker Time (min)'], errors='coerce')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Boxplot: Total Distance by Routing Policy
    sns.boxplot(x='Routing', y='Total Distance Walked (m)', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Total Distance Walked by Routing Policy')

    # Barplot: Average Picker Time by Batching Policy
    sns.barplot(x='Batching', y='Average Picker Time (min)', data=df, errorbar='sd', ax=axes[0, 1])
    axes[0, 1].set_title('Average Picker Time by Batching Policy')

    # Scatterplot: Distance vs Picker Time by Routing
    sns.scatterplot(x='Total Distance Walked (m)', y='Average Picker Time (min)', hue='Routing', data=df, ax=axes[1, 0])
    axes[1, 0].set_title('Distance vs. Picker Time by Routing')

    # Pivot Table: Average Distance by Routing Policy (as barplot)
    pivot = df.pivot_table(index='Routing', values='Total Distance Walked (m)', aggfunc='mean')
    pivot.plot(kind='bar', ax=axes[1, 1], legend=False)
    axes[1, 1].set_ylabel('Avg Total Distance Walked (m)')
    axes[1, 1].set_title('Average Total Distance by Routing Policy')

    plt.tight_layout()
    plt.show()

def make_affinity_slotting(sku_ids, locations, orders):
    sku_groups = group_skus_by_affinity(compute_affinity_matrix(orders), sku_ids, group_size=18)
    return AffinitySlotting(sku_groups)

def make_popularity_abc_slotting(sku_ids, locations, orders, sku_popularity):
    return PopularityABCSlotting(sku_popularity)

def make_random_slotting(sku_ids, locations, orders):
    return RandomSlotting()

def make_round_robin_slotting(sku_ids, locations, orders):
    return RoundRobinSlotting()

def make_greedy_batching():
    return BatchingPolicyAdapter(GreedyProximityBatching())

def make_seed_savings_batching():
    return BatchingPolicyAdapter(SeedSavingsBatching())

def make_round_robin_batching():
    return BatchingPolicyAdapter(RoundRobinBatching())

def make_random_batching():
    return BatchingPolicyAdapter(RandomBatching())

def make_side_grouped_routing():
    return SideGroupedRouting()

def make_s_shape_routing():
    return SShapeRouting()

def make_largest_gap_routing():
    return LargestGapRouting()

def make_hybrid_combined_routing():
    return HybridCombinedRouting()

if __name__ == "__main__":
    run_experiments()