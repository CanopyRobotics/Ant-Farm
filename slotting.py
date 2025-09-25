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