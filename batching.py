from typing import List
from models import Order
from collections import defaultdict
import numpy as np
from simulation import location_to_xy

class BatchingPolicy:
    def batch(self, orders: List[Order], num_pickers: int) -> List[List[Order]]:
        raise NotImplementedError

class RoundRobinBatching(BatchingPolicy):
    def batch(self, orders, num_pickers):
        buckets = [[] for _ in range(num_pickers)]
        for i, o in enumerate(orders):
            buckets[i % num_pickers].append(o)
        return buckets

class GreedyProximityBatching:
    def batch(self, orders, num_pickers, sku_to_location, wh, sections_per_side, max_orders_per_picker=25):
        batches = [[] for _ in range(num_pickers)]
        remaining_orders = orders.copy()

        # Precompute order centroids for proximity
        def order_centroid(order):
            xs, ys = [], []
            for line in order.lines:
                loc = sku_to_location[line.sku_id]
                x, y = location_to_xy(loc, wh, sections_per_side)
                xs.append(x)
                ys.append(y)
            return np.mean(xs), np.mean(ys)

        order_centroids = [order_centroid(order) for order in remaining_orders]

        for picker_idx in range(num_pickers):
            if not remaining_orders:
                break
            batch = []
            idx = 0
            batch.append(remaining_orders.pop(idx))
            batch_centroids = [order_centroids.pop(idx)]
            while len(batch) < max_orders_per_picker and remaining_orders:
                batch_x = np.mean([c[0] for c in batch_centroids])
                batch_y = np.mean([c[1] for c in batch_centroids])
                dists = [np.hypot(c[0] - batch_x, c[1] - batch_y) for c in order_centroids]
                next_idx = int(np.argmin(dists))
                batch.append(remaining_orders.pop(next_idx))
                batch_centroids.append(order_centroids.pop(next_idx))
            batches[picker_idx] = batch
        return batches

class SeedSavingsBatching:
    def batch(self, orders, num_pickers, sku_to_location, wh, sections_per_side, max_orders_per_picker=25):
        batches = [[] for _ in range(num_pickers)]
        remaining_orders = orders.copy()

        # Helper: Compute batch path length if a new order is added
        def incremental_distance(batch, candidate_order):
            # Get all pick locations in batch + candidate
            xs, ys = [], []
            for order in batch + [candidate_order]:
                for line in order.lines:
                    loc = sku_to_location[line.sku_id]
                    x, y = location_to_xy(loc, wh, sections_per_side)
                    xs.append(x)
                    ys.append(y)
            # Approximate path length by total distance between consecutive picks (nearest neighbor)
            points = list(zip(xs, ys))
            if not points:
                return 0
            # Start at I/O (0,0)
            current = (0, 0)
            unvisited = points.copy()
            total_dist = 0
            while unvisited:
                dists = [np.hypot(current[0] - p[0], current[1] - p[1]) for p in unvisited]
                idx = int(np.argmin(dists))
                total_dist += dists[idx]
                current = unvisited.pop(idx)
            # Return to I/O
            total_dist += np.hypot(current[0], current[1])
            return total_dist

        for picker_idx in range(num_pickers):
            if not remaining_orders:
                break
            # Seed: pick a random order (could use other heuristics)
            seed_idx = np.random.randint(len(remaining_orders))
            batch = [remaining_orders.pop(seed_idx)]
            while len(batch) < max_orders_per_picker and remaining_orders:
                # For each candidate, compute incremental distance if added
                current_dist = incremental_distance(batch, Order(id=-1, lines=[]))
                savings = []
                for i, order in enumerate(remaining_orders):
                    dist_with_order = incremental_distance(batch, order)
                    savings.append(current_dist - dist_with_order)  # Higher savings is better
                best_idx = int(np.argmax(savings))
                batch.append(remaining_orders.pop(best_idx))
            batches[picker_idx] = batch
        return batches

class RandomBatching(BatchingPolicy):
    def batch(self, orders, num_pickers):
        import random
        buckets = [[] for _ in range(num_pickers)]
        shuffled_orders = orders[:]
        random.shuffle(shuffled_orders)
        for i, o in enumerate(shuffled_orders):
            buckets[i % num_pickers].append(o)
        return buckets

class BatchingPolicyAdapter:
    def __init__(self, policy):
        self.policy = policy

    def batch(self, orders, num_pickers, sku_to_location=None, wh=None, sections_per_side=None, max_orders_per_picker=None):
        # RoundRobinBatching only needs orders and num_pickers
        if isinstance(self.policy, RoundRobinBatching):
            return self.policy.batch(orders, num_pickers)
        # GreedyProximityBatching needs all arguments, but provide defaults if missing
        elif isinstance(self.policy, GreedyProximityBatching):
            # Provide sensible defaults if not given
            if sku_to_location is None:
                sku_to_location = {}
            if wh is None:
                # Dummy warehouse config (adjust as needed)
                from models import WarehouseCfg
                wh = WarehouseCfg(num_aisles=10, aisle_length_m=20.0, aisle_width_m=3.0, speed_mps=0.75, pick_time_s=20.0)
            if sections_per_side is None:
                sections_per_side = 10
            if max_orders_per_picker is None:
                max_orders_per_picker = 25
            return self.policy.batch(orders, num_pickers, sku_to_location, wh, sections_per_side, max_orders_per_picker)
        # SeedSavingsBatching needs all arguments, but provide defaults if missing
        elif isinstance(self.policy, SeedSavingsBatching):
            if sku_to_location is None:
                sku_to_location = {}
            if wh is None:
                from models import WarehouseCfg
                wh = WarehouseCfg(num_aisles=10, aisle_length_m=20.0, aisle_width_m=3.0, speed_mps=0.75, pick_time_s=20.0)
            if sections_per_side is None:
                sections_per_side = 10
            if max_orders_per_picker is None:
                max_orders_per_picker = 25
            return self.policy.batch(orders, num_pickers, sku_to_location, wh, sections_per_side, max_orders_per_picker)
        # Add more elifs for other policies as needed
        else:
            # Default fallback: just use orders and num_pickers
            return self.policy.batch(orders, num_pickers)