from typing import List
from models import Order

class BatchingPolicy:
    def batch(self, orders: List[Order], num_pickers: int) -> List[List[Order]]:
        raise NotImplementedError

class RoundRobinBatching(BatchingPolicy):
    def batch(self, orders, num_pickers):
        buckets = [[] for _ in range(num_pickers)]
        for i, o in enumerate(orders):
            buckets[i % num_pickers].append(o)
        return buckets