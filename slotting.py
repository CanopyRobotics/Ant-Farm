from typing import List, Dict
from storage import StorageLocation

class SlottingPolicy:
    def assign(self, sku_ids: List[int], locations: List[StorageLocation]) -> Dict[int, StorageLocation]:
        raise NotImplementedError

class RandomSlotting(SlottingPolicy):
    def assign(self, sku_ids, locations):
        import random
        n = len(locations)
        rng = random.Random()
        return {sku: locations[rng.randrange(n)] for sku in sku_ids}

class RoundRobinSlotting(SlottingPolicy):
    def assign(self, sku_ids, locations):
        n = len(locations)
        return {sku: locations[i % n] for i, sku in enumerate(sku_ids)}

class PopularityABCSlotting(SlottingPolicy):
    def __init__(self, sku_popularity: Dict[int, float]):
        self.sku_popularity = sku_popularity  # e.g., {sku_id: frequency}

    def assign(self, sku_ids: List[int], locations: List[StorageLocation]):
        # Sort SKUs by popularity (descending)
        sorted_skus = sorted(sku_ids, key=lambda sku: -self.sku_popularity.get(sku, 0))
        # Sort locations by proximity (implement your own logic if needed)
        # For now, just use the order in the locations list as "closest to farthest"
        # For ABC: assign A to first N, B to next, C to rest
        n = len(sorted_skus)
        a_cut = int(0.2 * n)
        b_cut = int(0.5 * n)
        # Optionally, you could use this info for reporting or coloring in the visualization

        # Assign most popular SKUs to closest locations
        assignment = {}
        for i, sku in enumerate(sorted_skus):
            assignment[sku] = locations[i % len(locations)]
        return assignment