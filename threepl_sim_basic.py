#!/usr/bin/env python3
"""
threepl_sim_basic.py
A from-scratch, single-file Python script to simulate an e-commerce 3PL picker-to-part warehouse.

Features (beginner-friendly):
- Multiple pickers
- Aisle blocking (capacity = 1)
- Simple storage policies: random | popularity_near_io
- Basic KPIs (mean/p50/p90 order time, travel vs pick time, makespan, throughput)
- Zero dependencies (standard library only)

Run examples:
  python threepl_sim_basic.py
  python threepl_sim_basic.py --num-pickers 4 --num-orders 300 --policy popularity_near_io
  python threepl_sim_basic.py --num-aisles 12 --racks-per-aisle 50 --order-lines-mean 3

Tip: Start with defaults, then tweak one flag at a time.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import argparse
import math
import random
import statistics
import sys
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
import random
from collections import defaultdict

# -------------------- Domain models --------------------
@dataclass(frozen=True) # Represents one line in an order (1 SKU, qty)
class OrderLine: 
    sku_id: int
    qty: int = 1

@dataclass(frozen=True)
class Order: # Represents one order (multiple lines)
    id: int
    lines: List[OrderLine]

@dataclass
class WarehouseCfg: # Warehouse configuration parameters
    num_aisles: int = 10
    aisle_length_m: float = 75.0
    aisle_width_m: float = 3.0
    speed_mps: float = 0.75         # walking speed (m/s) with cart
    pick_time_s: float = 20.0       # seconds per pick

@dataclass
class RunCfg: # Simulation run parameters
    num_pickers: int = 3
    num_orders: int = 200
    order_lines_mean: float = 3.0   # average lines per order (Poisson-like)
    seed: int = 42
    policy: str = "popularity_near_io"  # or "random"

# -------------------- Helpers --------------------
def travel_time_across(cfg: WarehouseCfg) -> float:
    """Time to cross an aisle (perpendicular move)."""
    return cfg.aisle_width_m / cfg.speed_mps

def travel_time_in_aisle(cfg: WarehouseCfg) -> float:
    """Time to traverse down an aisle (simplified: full length)."""
    return cfg.aisle_length_m / cfg.speed_mps

def poisson_like(mean: float, rng: random.Random) -> int:
    """Simple Poisson-like sampler using sum of exponentials method (no numpy)."""
    # Knuth's algorithm (works fine for small means)
    L = math.exp(-mean)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return max(1, k - 1)

# -------------------- Storage policies --------------------
def assign_storage_random(sku_ids: List[int], num_aisles: int, rng: random.Random) -> Dict[int, int]:
    """sku_id -> aisle (1..num_aisles), random uniform."""
    return {sku: (rng.randrange(num_aisles) + 1) for sku in sku_ids}

def assign_storage_popularity_near_io(sku_ids: List[int], num_aisles: int) -> Dict[int, int]:
    """
    Greedy near-I/O: more 'popular' SKUs (we fake popularity by id ascending) get earlier aisles.
    This fills aisle 1, then 2, etc., round-robin across aisles for simplicity.
    """
    layout: Dict[int, int] = {}
    idx = 0
    for sku in sku_ids:  # sku_ids already in ascending order
        layout[sku] = (idx % num_aisles) + 1
        idx += 1
    return layout

# -------------------- New Storage Location System --------------------

class AisleSide(Enum):
    LEFT = "L"
    RIGHT = "R"

@dataclass(frozen=True)
class StorageLocation:
    side_id: str              # e.g., "A003"
    section: str              # e.g., "A003-AB"
    tote_id: str              # e.g., "A003-AB-032"

    def __str__(self):
        return self.tote_id

    @property
    def aisle(self) -> int:
        # Extract aisle number from side_id: "A003" -> 2 for left, 2 for right (A003/A004 both aisle 2)
        # left side: odd, right side: even
        side_num = int(self.side_id[1:])
        return (side_num + 1) // 2

def gen_storage_locations(
    num_aisles: int,
    sections_per_side: int
) -> List[StorageLocation]:
    locations = []
    # Side numbering: left = A001, right = A002, next aisle left = A003, right = A004, etc.
    for aisle in range(1, num_aisles + 1):
        left_side_id = f"A{2*aisle-1:03d}"
        right_side_id = f"A{2*aisle:03d}"
        for side_id in [left_side_id, right_side_id]:
            for s_idx in range(sections_per_side):
                # Section letters: AA, AB, AC, ...
                section_letters = chr(ord('A') + (s_idx // 26)) + chr(ord('A') + (s_idx % 26))
                section_id = f"{side_id}-{section_letters}"
                for level in range(1, 7):  # 6 levels, top=1, bottom=6
                    for tote in range(1, 4):  # 3 totes per level
                        # Tote numbering: top left = 001, top middle = 002, top right = 003, next level = 011, 012, 013, ..., bottom = 051, 052, 053
                        tote_num = (level - 1) * 10 + tote
                        tote_id = f"{section_id}-{tote_num:03d}"
                        locations.append(StorageLocation(
                            side_id=side_id,
                            section=section_id,
                            tote_id=tote_id
                        ))
    return locations

# Example: Assign each SKU to a storage location (round-robin)
def assign_skus_to_locations(sku_ids: List[int], locations: List[StorageLocation]) -> Dict[int, StorageLocation]:
    mapping = {}
    n = len(locations)
    for idx, sku in enumerate(sku_ids):
        mapping[sku] = locations[idx % n]
    return mapping

# -------------------- Data generation --------------------
def gen_skus(num_aisles: int, sections_per_side: int) -> List[int]:
    """Create a list of sku_ids sized to fill all storage locations."""
    num_locations = num_aisles * 2 * sections_per_side * 6 * 3
    return list(range(1, num_locations + 1))

def gen_orders(num_orders: int, sku_ids: List[int], mean_lines: float, rng: random.Random) -> List[Order]:
    """Generate orders with random SKUs per line, fully stocked warehouse (with replacement)."""
    orders: List[Order] = []
    n = len(sku_ids)
    for oid in range(1, num_orders + 1):
        lines = max(1, poisson_like(mean_lines, rng))
        chosen = [rng.choice(sku_ids) for _ in range(lines)]
        orders.append(Order(id=oid, lines=[OrderLine(sku_id=s) for s in chosen]))
    return orders

def split_orders_round_robin(orders: List[Order], num_pickers: int) -> List[List[Order]]:
    """Assign orders to pickers in a simple round-robin fashion."""
    buckets = [[] for _ in range(num_pickers)]
    for i, o in enumerate(orders):
        buckets[i % num_pickers].append(o)
    return buckets

# -------------------- Simulation core (updated to use StorageLocation) --------------------
def simulate_with_blocking_locations(
    wh: WarehouseCfg,
    orders_by_picker: List[List[Order]],
    sku_to_location: Dict[int, StorageLocation]
) -> Dict[str, float]:
    """
    Multi-picker simulation with aisle blocking, using full storage locations.
    - Each aisle has capacity 1. If busy, a picker must wait.
    - For each order, group items by aisle, and traverse aisles in order.
    - Within each aisle, group by side/section/level/tote if needed (future extension).
    """
    next_free_time: Dict[int, float] = {a: 0.0 for a in range(1, wh.num_aisles + 1)}
    picker_times: List[float] = [0.0 for _ in orders_by_picker]
    all_order_durations: List[float] = []
    all_travel_times: List[float] = []
    all_pick_times: List[float] = []

    for p_idx, order_list in enumerate(orders_by_picker):
        t = picker_times[p_idx]
        for o in order_list:
            start = t
            travel_t = 0.0
            pick_t = 0.0

            # Group items by aisle (using StorageLocation)
            items_by_aisle: Dict[int, List[StorageLocation]] = {}
            for ln in o.lines:
                loc = sku_to_location[ln.sku_id]
                items_by_aisle.setdefault(loc.aisle, []).append(loc)

            # Traverse aisles in order
            for a in range(1, wh.num_aisles + 1):
                if a not in items_by_aisle:
                    # Still cross the aisle to move forward
                    dt = travel_time_across(wh)
                    t += dt
                    travel_t += dt
                    continue

                # Always cross to move forward
                dt = travel_time_across(wh)
                t += dt
                travel_t += dt

                # Wait for aisle to be free (blocking)
                if t < next_free_time[a]:
                    t = next_free_time[a]

                # Traverse full aisle (simplified)
                dt = travel_time_in_aisle(wh)
                t += dt
                travel_t += dt

                # Pick items (all items in this aisle)
                picks_here = len(items_by_aisle[a])
                dt = picks_here * wh.pick_time_s
                t += dt
                pick_t += dt

                # Release aisle
                next_free_time[a] = t

            end = t
            all_order_durations.append(end - start)
            all_travel_times.append(travel_t)
            all_pick_times.append(pick_t)

        picker_times[p_idx] = t  # final time for this picker

    # KPIs
    durations = all_order_durations
    travel = all_travel_times
    pick = all_pick_times

    if not durations:
        return {}

    try:
        p90 = float(statistics.quantiles(durations, n=10)[8])
    except Exception:
        p90 = max(durations)

    makespan = max(picker_times)
    throughput_per_hour = (len(durations) / makespan) * 3600.0 if makespan > 0 else 0.0

    return {
        "orders": float(len(durations)),
        "num_pickers": float(len(orders_by_picker)),
        "mean_order_time_s": statistics.fmean(durations),
        "p50_order_time_s": statistics.median(durations),
        "p90_order_time_s": p90,
        "mean_travel_time_s": statistics.fmean(travel),
        "mean_pick_time_s": statistics.fmean(pick),
        "travel_fraction": (sum(travel) / (sum(travel) + sum(pick))) if (sum(travel)+sum(pick)) > 0 else 0.0,
        "makespan_s": makespan,
        "throughput_orders_per_hour": throughput_per_hour,
    }

# -------------------- CLI & main --------------------
def parse_args(argv: List[str]) -> Tuple[WarehouseCfg, RunCfg, int]:
    parser = argparse.ArgumentParser(description="Basic 3PL picker-to-part simulation (from scratch).")
    parser.add_argument("--num-aisles", type=int, default=10, help="Number of aisles")
    parser.add_argument("--sections-per-side", type=int, default=10, help="Rack sections per aisle side (A, B, ...)")
    parser.add_argument("--num-pickers", type=int, default=3, help="Number of pickers")
    parser.add_argument("--num-orders", type=int, default=200, help="Number of orders to simulate")
    parser.add_argument("--order-lines-mean", type=float, default=3.0, help="Mean order lines (Poisson-like)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--policy", type=str, default="popularity_near_io", help="Storage policy")
    parser.add_argument("--aisle-length-m", type=float, default=20.0, help="Aisle length (meters)")
    parser.add_argument("--aisle-width-m", type=float, default=3.0, help="Aisle width (meters)")
    parser.add_argument("--speed-mps", type=float, default=0.75, help="Walking speed (m/s)")
    parser.add_argument("--pick-time-s", type=float, default=20.0, help="Seconds per pick")
    args = parser.parse_args(argv)

    wh = WarehouseCfg(
        num_aisles=args.num_aisles,
        aisle_length_m=args.aisle_length_m,
        aisle_width_m=args.aisle_width_m,
        speed_mps=args.speed_mps,
        pick_time_s=args.pick_time_s,
    )
    run = RunCfg(
        num_pickers=args.num_pickers,
        num_orders=args.num_orders,
        order_lines_mean=args.order_lines_mean,
        seed=args.seed,
        policy=args.policy,
    )
    return wh, run, args.sections_per_side

def print_warehouse_layout(num_aisles: int, sections_per_side: int):
    print("\n=== Warehouse Layout ===")
    for aisle in range(1, num_aisles + 1):
        left_side_id = f"A{2*aisle-1:03d}"
        right_side_id = f"A{2*aisle:03d}"
        print(f"\nAisle {aisle}:")
        for side, side_id in [("Left", left_side_id), ("Right", right_side_id)]:
            print(f"  {side} ({side_id}):")
            for s_idx in range(sections_per_side):
                section_letters = chr(ord('A') + (s_idx // 26)) + chr(ord('A') + (s_idx % 26))
                section_id = f"{side_id}-{section_letters}"
                print(f"    Section {section_letters} ({section_id}): ", end="")
                tote_ids = []
                for level in range(1, 7):
                    for tote in range(1, 4):
                        tote_num = (level - 1) * 10 + tote
                        tote_id = f"{section_id}-{tote_num:03d}"
                        tote_ids.append(tote_id)
                # Show first and last 3 totes for brevity
                print(", ".join(tote_ids[:3]), "...", ", ".join(tote_ids[-3:]))

def legal_warehouse_distance(p1, p2, aisle_xs, total_height, cross_aisle_height):
    # p1, p2: (x, y) rack locations
    # aisle_xs: list of aisle center x positions
    # total_height: height of racks
    # cross_aisle_height: height of cross-aisle
    # The picker can use top or bottom cross-aisle for each pick
    # Compute distance for both options and take the minimum

    x1, y1 = p1
    x2, y2 = p2

    # Find nearest aisle center for each pick
    aisle_x1 = min(aisle_xs, key=lambda x: abs(x - x1))
    aisle_x2 = min(aisle_xs, key=lambda x: abs(x - x2))

    # Bottom cross-aisle y
    y_bot = -cross_aisle_height / 2
    # Top cross-aisle y
    y_top = total_height + cross_aisle_height / 2

    # Option 1: use bottom cross-aisle
    d1 = abs(y1 - y_bot) + abs(aisle_x1 - aisle_x2) + abs(y2 - y_bot)
    # Option 2: use top cross-aisle
    d2 = abs(y1 - y_top) + abs(aisle_x1 - aisle_x2) + abs(y2 - y_top)
    return min(d1, d2)

def build_legal_picker_path(order_points, start_point, aisle_xs, total_height, cross_aisle_height):
    N = len(order_points)
    dist_matrix = [[0.0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                dist_matrix[i][j] = legal_warehouse_distance(
                    order_points[i], order_points[j], aisle_xs, total_height, cross_aisle_height
                )
    unvisited = set(range(1, N))
    path_order = [0]
    current = 0
    while unvisited:
        nearest = min(unvisited, key=lambda idx: dist_matrix[current][idx])
        path_order.append(nearest)
        current = nearest
        unvisited.remove(nearest)
    # Do NOT return to start

    picker_path = []
    for i in range(len(path_order)-1):
        a = order_points[path_order[i]]
        b = order_points[path_order[i+1]]
        x1, y1 = a
        x2, y2 = b
        # Find nearest aisle center for each pick
        aisle_x1 = min(aisle_xs, key=lambda x: abs(x - x1))
        aisle_x2 = min(aisle_xs, key=lambda x: abs(x - x2))
        if abs(aisle_x1 - aisle_x2) < 1e-6:
            # Same aisle: move directly up/down the aisle
            if not picker_path or picker_path[-1] != (x1, y1):
                picker_path.append((x1, y1))
            if (x1, y1) != (x2, y2):
                picker_path.append((x2, y2))
        else:
            # Different aisle: use cross-aisle (choose top or bottom)
            y_bot = -cross_aisle_height / 2
            y_top = total_height + cross_aisle_height / 2
            # Option 1: bottom cross-aisle
            d1 = abs(y1 - y_bot) + abs(aisle_x1 - aisle_x2) + abs(y2 - y_bot)
            # Option 2: top cross-aisle
            d2 = abs(y1 - y_top) + abs(aisle_x1 - aisle_x2) + abs(y2 - y_top)
            if d1 <= d2:
                # Use bottom cross-aisle
                if not picker_path or picker_path[-1] != (x1, y1):
                    picker_path.append((x1, y1))
                if picker_path[-1] != (aisle_x1, y_bot):
                    picker_path.append((aisle_x1, y_bot))
                if (aisle_x1, y_bot) != (aisle_x2, y_bot):
                    picker_path.append((aisle_x2, y_bot))
                if (aisle_x2, y_bot) != (x2, y_bot):
                    picker_path.append((x2, y_bot))
                if (x2, y_bot) != (x2, y2):
                    picker_path.append((x2, y2))
            else:
                # Use top cross-aisle
                if not picker_path or picker_path[-1] != (x1, y1):
                    picker_path.append((x1, y1))
                if picker_path[-1] != (aisle_x1, y_top):
                    picker_path.append((aisle_x1, y_top))
                if (aisle_x1, y_top) != (aisle_x2, y_top):
                    picker_path.append((aisle_x2, y_top))
                if (aisle_x2, y_top) != (x2, y_top):
                    picker_path.append((x2, y_top))
                if (x2, y_top) != (x2, y2):
                    picker_path.append((x2, y2))
    return picker_path

def build_serpentine_picker_path(pick_points, start_point, aisle_xs, total_height, cross_aisle_height):
    # pick_points: list of (x, y, aisle_idx) for all picks
    # Group picks by aisle
    from collections import defaultdict
    aisle_picks = defaultdict(list)
    for x, y, aisle_idx in pick_points:
        aisle_picks[aisle_idx].append((x, y))

    # Compute aisle entrances (bottom center of each aisle)
    aisle_entrances = {a: (aisle_xs[a], -cross_aisle_height / 2) for a in aisle_picks}

    # Build a list of aisles to visit
    aisles_to_visit = list(aisle_picks.keys())

    # Nearest neighbor TSP on aisle entrances (start at aisle 1)
    unvisited = set(aisles_to_visit)
    current = min(aisle_entrances, key=lambda a: abs(aisle_entrances[a][0] - start_point[0]))
    aisle_order = [current]
    unvisited.remove(current)
    while unvisited:
        last_entrance = aisle_entrances[current]
        next_aisle = min(unvisited, key=lambda a: abs(aisle_entrances[a][0] - last_entrance[0]))
        aisle_order.append(next_aisle)
        current = next_aisle
        unvisited.remove(current)

    # Build the picker path
    picker_path = [start_point]
    for aisle_idx in aisle_order:
        picks = aisle_picks[aisle_idx]
        # Sort picks by y (section position)
        picks_sorted = sorted(picks, key=lambda p: p[1])
        # Decide direction: up (bottom to top) or down (top to bottom)
        # We'll always go bottom to top, then top to bottom if needed
        bottom = (aisle_xs[aisle_idx], -cross_aisle_height / 2)
        top = (aisle_xs[aisle_idx], total_height + cross_aisle_height / 2)
        # Move to aisle entrance (bottom)
        if picker_path[-1] != bottom:
            picker_path.append(bottom)
        # Walk up the aisle, picking as you go
        for x, y in picks_sorted:
            if picker_path[-1] != (x, y):
                picker_path.append((x, y))
        # Move to top cross-aisle
        if picker_path[-1][1] != total_height + cross_aisle_height / 2:
            picker_path.append((aisle_xs[aisle_idx], total_height + cross_aisle_height / 2))
        # (Optional: If you want to allow walking down for more picks, add logic here)

    return picker_path

def build_side_grouped_picker_path(pick_points, start_point, side_xs, total_height, cross_aisle_height):
    # pick_points: list of (x, y, side_id) for all picks
    from collections import defaultdict
    side_picks = defaultdict(list)
    for x, y, side_id in pick_points:
        side_picks[side_id].append((x, y))

    # Sort side_ids in order (A001, A002, ...)
    sorted_sides = sorted(side_picks.keys(), key=lambda s: int(s[1:]))

    y_bot = -cross_aisle_height / 2
    y_top = total_height + cross_aisle_height / 2

    picker_path = [start_point]
    current_pos = start_point

    for side_id in sorted_sides:
        picks = side_picks[side_id]
        picks_sorted = sorted(picks, key=lambda p: p[1])
        x_side = side_xs[side_id]

        # Decide whether to enter from bottom or top cross-aisle, based on which is closer to current_pos
        dist_to_bot = abs(current_pos[1] - y_bot)
        dist_to_top = abs(current_pos[1] - y_top)
        if dist_to_bot <= dist_to_top:
            entry_y = y_bot
        else:
            entry_y = y_top

        # Move horizontally in the cross-aisle to the new side if needed
        if current_pos[1] != entry_y:
            picker_path.append((current_pos[0], entry_y))
        if current_pos[0] != x_side:
            picker_path.append((x_side, entry_y))

        # Decide pick direction: up or down
        if entry_y == y_bot:
            picks_in_order = sorted(picks, key=lambda p: p[1])
        else:
            picks_in_order = sorted(picks, key=lambda p: -p[1])

        # Walk along the side, picking as you go
        for x, y in picks_in_order:
            if picker_path[-1] != (x, y):
                picker_path.append((x, y))

        # After finishing this side, update current_pos to the last pick
        current_pos = picker_path[-1]

    return picker_path
# -------------------- Visualization --------------------
def plot_warehouse_map(num_aisles: int, sections_per_side: int, wh: WarehouseCfg, orders_by_picker=None, sku_to_location=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.widgets import Slider

    # Physical dimensions
    aisle_width = wh.aisle_width_m
    aisle_length = wh.aisle_length_m
    rack_depth = 0.5
    section_length = aisle_length / sections_per_side
    cross_aisle_height = 2.0  # meters

    # Compute X positions for each aisle and rack
    x_positions = []
    for aisle in range(num_aisles):
        x_left_rack = aisle * (2 * rack_depth + aisle_width)
        x_aisle = x_left_rack + rack_depth
        x_right_rack = x_aisle + aisle_width
        x_positions.append((x_left_rack, x_aisle, x_right_rack))

    total_height = sections_per_side * section_length

    # Precompute aisle center x positions
    aisle_xs = [x[1] + aisle_width / 2 for x in x_positions]

    # Prepare globally optimized picker path (all picks in one route, using legal warehouse walking)
    picker_path = []
    if orders_by_picker and sku_to_location:
        picker_orders = orders_by_picker[0]
        pick_points = []
        side_xs = {}
        for o in picker_orders:
            for ln in o.lines:
                loc = sku_to_location[ln.sku_id]
                aisle_idx = loc.aisle - 1
                is_left = int(loc.side_id[-1]) % 2 == 1
                section_letters = loc.section.split('-')[-1]
                s_idx = (ord(section_letters[0]) - ord('A')) * 26 + (ord(section_letters[1]) - ord('A'))
                y = s_idx * section_length + section_length / 2
                if is_left:
                    x = x_positions[aisle_idx][0] + rack_depth / 2
                else:
                    x = x_positions[aisle_idx][2] + rack_depth / 2
                pick_points.append((x, y, loc.side_id))
                side_xs[loc.side_id] = x
        start_point = (x_positions[0][1] + aisle_width / 2, -cross_aisle_height / 2)
        picker_path = build_side_grouped_picker_path(pick_points, start_point, side_xs, total_height, cross_aisle_height)

    # Set up plot
    fig, ax = plt.subplots(figsize=(num_aisles * 2.5, sections_per_side * 1.2))
    plt.subplots_adjust(bottom=0.18)

    # Draw racks and labels
    for aisle in range(num_aisles):
        x_left_rack, x_aisle, x_right_rack = x_positions[aisle]
        aisle_num = aisle + 1
        left_side_id = f"A{2*aisle_num-1:03d}"
        right_side_id = f"A{2*aisle_num:03d}"

        # Draw left rack sections
        for s_idx in range(sections_per_side):
            section_letters = chr(ord('A') + (s_idx // 26)) + chr(ord('A') + (s_idx % 26))
            y = s_idx * section_length
            rect = patches.Rectangle((x_left_rack, y), rack_depth, section_length, linewidth=1, edgecolor='black', facecolor='lightblue')
            ax.add_patch(rect)
            ax.text(x_left_rack + rack_depth / 2, y + section_length / 2, section_letters, ha='center', va='center', fontsize=8)

        # Draw right rack sections
        for s_idx in range(sections_per_side):
            section_letters = chr(ord('A') + (s_idx // 26)) + chr(ord('A') + (s_idx % 26))
            y = s_idx * section_length
            rect = patches.Rectangle((x_right_rack, y), rack_depth, section_length, linewidth=1, edgecolor='black', facecolor='lightgreen')
            ax.add_patch(rect)
            ax.text(x_right_rack + rack_depth / 2, y + section_length / 2, section_letters, ha='center', va='center', fontsize=8)

        # Place labels above the top cross-aisle
        label_y = total_height + cross_aisle_height + 0.5

        # Aisle label
        ax.text(
            x_aisle + aisle_width / 2,
            label_y,
            f"Aisle {aisle_num}",
            ha='center', va='bottom', fontsize=10, color='navy', rotation=90
        )
        # Side labels
        ax.text(
            x_left_rack + rack_depth / 2,
            label_y + 0.5,
            left_side_id,
            ha='center', va='bottom', fontsize=9, color='blue', fontweight='bold', rotation=90
        )
        ax.text(
            x_right_rack + rack_depth / 2,
            label_y + 0.5,
            right_side_id,
            ha='center', va='bottom', fontsize=9, color='green', fontweight='bold', rotation=90
        )

    # Draw cross-aisle at bottom
    ax.add_patch(
        patches.Rectangle(
            (x_positions[0][0] - rack_depth, -cross_aisle_height),
            x_positions[-1][2] + rack_depth - (x_positions[0][0] - rack_depth),
            cross_aisle_height,
            linewidth=0,
            facecolor='#f0e68c',
            alpha=0.5,
            zorder=0
        )
    )
    ax.text(x_positions[0][0] + 1, -cross_aisle_height / 2, "Cross-aisle", va='center', ha='left', fontsize=10, color='brown')

    # Draw cross-aisle at top
    ax.add_patch(
        patches.Rectangle(
            (x_positions[0][0] - rack_depth, total_height),
            x_positions[-1][2] + rack_depth - (x_positions[0][0] - rack_depth),
            cross_aisle_height,
            linewidth=0,
            facecolor='#f0e68c',
            alpha=0.5,
            zorder=0
        )
    )
    ax.text(x_positions[0][0] + 1, total_height + cross_aisle_height / 2, "Cross-aisle", va='center', ha='left', fontsize=10, color='brown')

    ax.set_xlim(
        x_positions[0][0] - rack_depth - 0.5,
        x_positions[-1][2] + rack_depth + 0.5
    )
    ax.set_ylim(-cross_aisle_height - 0.5, total_height + cross_aisle_height + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("Warehouse Map (meters): Racks, Aisles, Sides, Picker Path", fontsize=14)

    # Add slider
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.04])
    slider = Slider(ax_slider, 'Step', 1, max(1, len(picker_path)), valinit=1, valstep=1)

    # Draw initial picker path
    path_line, = ax.plot([], [], '-', color='red', linewidth=2, alpha=0.7)
    path_dots, = ax.plot([], [], 'o', color='red', markersize=5, alpha=0.7)

    def update(val):
        step = int(slider.val)
        xs = [p[0] for p in picker_path[:step]]
        ys = [p[1] for p in picker_path[:step]]
        path_line.set_data(xs, ys)
        path_dots.set_data(xs, ys)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(1)

    plt.show()
    return picker_path

if __name__ == "__main__":
    wh = WarehouseCfg(num_aisles=10, aisle_length_m=20.0, aisle_width_m=3.0)
    sections_per_side = 10
    run = RunCfg(num_pickers=1, num_orders=25, order_lines_mean=4, seed=42)
    rng = random.Random(run.seed)
    sku_ids = gen_skus(wh.num_aisles, sections_per_side)
    locations = gen_storage_locations(wh.num_aisles, sections_per_side)
    sku_to_location = assign_skus_to_locations(sku_ids, locations)
    orders = gen_orders(run.num_orders, sku_ids, run.order_lines_mean, rng)
    orders_by_picker = split_orders_round_robin(orders, run.num_pickers)
    plot_warehouse_map(wh.num_aisles, sections_per_side, wh, orders_by_picker, sku_to_location)