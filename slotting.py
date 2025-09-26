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

        # --- Sort locations by proximity to I/O point (0,0) ---
        def location_distance(loc):
            # Compute (x, y) for loc based on warehouse geometry
            aisle_idx = (int(loc.side_id[1:]) - 1) // 2
            is_left = int(loc.side_id[-1]) % 2 == 1
            rack_depth = 0.5
            aisle_width = 3.0
            section_length = 20.0 / 10
            x_left_rack = aisle_idx * (2 * rack_depth + aisle_width)
            x_aisle = x_left_rack + rack_depth
            x_right_rack = x_aisle + aisle_width
            x = x_left_rack if is_left else x_right_rack
            section_letters = loc.section.split('-')[-1]
            s_idx = (ord(section_letters[0]) - ord('A')) * 26 + (ord(section_letters[1]) - ord('A'))
            y = s_idx * section_length + section_length / 2
            return (x**2 + y**2)**0.5

        locations_sorted = sorted(locations, key=location_distance)
        # ------------------------------------------------------

        assignment = {}
        for i, sku in enumerate(sorted_skus):
            assignment[sku] = locations_sorted[i % len(locations_sorted)]
        return assignment