from typing import List
from models import Order
from collections import defaultdict

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
        import numpy as np
        batches = [[] for _ in range(num_pickers)]
        remaining_orders = orders.copy()

        # Precompute order centroids for proximity
        def order_centroid(order):
            xs, ys = [], []
            for line in order.lines:
                loc = sku_to_location[line.sku_id]
                from simulation import location_to_xy
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