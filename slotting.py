from typing import List, Dict, Optional
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
        # Sort locations by proximity to I/O (e.g., (0,0))
        def location_distance(loc):
            # You need to compute (x, y) for each location based on your warehouse geometry
            # Example for left/right racks:
            aisle_idx = (int(loc.side_id[1:]) - 1) // 2
            is_left = int(loc.side_id[-1]) % 2 == 1
            rack_depth = 0.5
            aisle_width = 3.0  # or use wh.aisle_width_m
            section_length = 20.0 / 10  # or use wh.aisle_length_m / sections_per_side
            x_left_rack = aisle_idx * (2 * rack_depth + aisle_width)
            x_aisle = x_left_rack + rack_depth
            x_right_rack = x_aisle + aisle_width
            x = x_left_rack if is_left else x_right_rack
            section_letters = loc.section.split('-')[-1]
            s_idx = (ord(section_letters[0]) - ord('A')) * 26 + (ord(section_letters[1]) - ord('A'))
            y = s_idx * section_length + section_length / 2
            return (x**2 + y**2)**0.5
        locations_sorted = sorted(locations, key=location_distance)
        assignment = {}
        for i, sku in enumerate(sorted_skus):
            assignment[sku] = locations_sorted[i % len(locations_sorted)]
        return assignment

class AffinitySlotting(SlottingPolicy):
    def __init__(self, sku_groups: List[List[int]], sku_popularity: Optional[Dict[int, float]] = None):
        self.sku_groups = sku_groups
        self.sku_popularity = sku_popularity or {}

    def assign(self, sku_ids: List[int], locations: List[StorageLocation]) -> Dict[int, StorageLocation]:
        assignment: Dict[int, StorageLocation] = {}

        # Helpers
        def _aisle_from_side(side_id: str) -> int:
            try:
                num = int(str(side_id)[1:])
                return (num + 1) // 2
            except Exception:
                return 0

        def _is_left(side_id: str) -> bool:
            try:
                num = int(str(side_id)[1:])
                return num % 2 == 1
            except Exception:
                return False

        def _section_idx(section: str) -> int:
            try:
                letters = str(section).split('-')[-1]
                a = ord(letters[0]) - ord('A')
                b = ord(letters[1]) - ord('A')
                return a * 26 + b
            except Exception:
                return 0

        def _section_center_xy(side_id: str, section: str) -> tuple[float, float]:
            rack_depth = 1.0
            aisle_width = 2.0
            # estimate section_length from max index present
            max_idx = max((_section_idx(loc.section) for loc in locations), default=0)
            sections_per_side_guess = max(1, max_idx + 1)
            section_length = 20.0 / sections_per_side_guess
            aisle = _aisle_from_side(side_id)
            left = _is_left(side_id)
            x_left_rack = (aisle - 1) * (2 * rack_depth + aisle_width)
            x_aisle = x_left_rack + rack_depth
            x_right_rack = x_aisle + aisle_width
            x = x_left_rack if left else x_right_rack
            s_idx = _section_idx(section)
            y = s_idx * section_length + section_length / 2.0
            return x, y

        def _euclid(p):
            return (p[0]*p[0] + p[1]*p[1]) ** 0.5

        # Build section buckets: (side_id, section) -> [locations]
        from collections import defaultdict
        section_buckets: Dict[tuple, List[StorageLocation]] = defaultdict(list)
        for loc in locations:
            section_buckets[(loc.side_id, loc.section)].append(loc)

        # Canonical tote order within a section (increasing numeric suffix)
        def _tote_num(tid: str) -> int:
            try:
                return int(str(tid).split('-')[-1])
            except Exception:
                return 0
        for key in section_buckets:
            section_buckets[key].sort(key=lambda l: _tote_num(l.tote_id))

        # Order sections by distance to I/O (pack/ship) ~ near aisle 1, bottom
        sections_ordered = sorted(section_buckets.keys(), key=lambda k: _euclid(_section_center_xy(k[0], k[1])))

        # Section capacity (totes per section); assume constant -> use mode
        caps = [len(v) for v in section_buckets.values()]
        if not caps:
            return assignment
        # mode or median
        try:
            from statistics import mode
            section_capacity = mode(caps)
        except Exception:
            section_capacity = int(sorted(caps)[len(caps)//2])

        # Initial assignment: map each group (cluster) to the next nearest section(s)
        groups = [list(g) for g in self.sku_groups]
        # sort groups by popularity weight descending so heavier groups get nearer sections
        def _group_weight(g: List[int]) -> float:
            if self.sku_popularity:
                return float(sum(self.sku_popularity.get(s, 0.0) for s in g))
            return float(len(g))
        groups_sorted = sorted(groups, key=_group_weight, reverse=True)

        # Track which sections are used and offsets
        sec_offsets: Dict[tuple, int] = {k: 0 for k in sections_ordered}
        sec_idx = 0
        group_to_sections: Dict[int, List[tuple]] = {}

        for gi, group in enumerate(groups_sorted):
            remaining = len(group)
            assigned_sections: List[tuple] = []
            while remaining > 0 and sec_idx < len(sections_ordered):
                skey = sections_ordered[sec_idx]
                used = sec_offsets[skey]
                capacity = len(section_buckets[skey])
                avail = capacity - used
                if avail <= 0:
                    sec_idx += 1
                    continue
                take = min(remaining, avail)
                # assign SKUs in this slice to this section slice
                assigned_sections.append(skey)
                remaining -= take
                sec_offsets[skey] += take
                # if section filled, move to next section for spillover
                if sec_offsets[skey] >= capacity:
                    sec_idx += 1
            group_to_sections[gi] = assigned_sections

        # Hill-climb: swap group-section assignments to reduce weighted distance (1-1 case prioritized)
        # Only meaningful when group size ~= section_capacity (contiguous section placement)
        if all(len(g) >= 1 for g in groups_sorted):
            # Build a list of candidate pairs where groups map to exactly one section
            single_assign = [(gi, secs[0]) for gi, secs in group_to_sections.items() if len(secs) == 1]
            # Precompute section distances
            sec_dist = {skey: _euclid(_section_center_xy(skey[0], skey[1])) for skey in sections_ordered}
            # Simple swap-based hill climb
            improved = True
            iters = 0
            while improved and iters < 200:
                improved = False
                iters += 1
                for i in range(len(single_assign)):
                    for j in range(i+1, len(single_assign)):
                        gi1, s1 = single_assign[i]
                        gi2, s2 = single_assign[j]
                        w1 = _group_weight(groups_sorted[gi1])
                        w2 = _group_weight(groups_sorted[gi2])
                        cur = w1*sec_dist[s1] + w2*sec_dist[s2]
                        alt = w1*sec_dist[s2] + w2*sec_dist[s1]
                        if alt + 1e-9 < cur:
                            # swap
                            single_assign[i] = (gi1, s2)
                            single_assign[j] = (gi2, s1)
                            group_to_sections[gi1] = [s2]
                            group_to_sections[gi2] = [s1]
                            improved = True
                            break
                    if improved:
                        break

        # Materialize final SKU->Location mapping from group_to_sections
        # Build iterators per section over locations
        sec_iters = {k: iter(section_buckets[k]) for k in section_buckets}
        for gi, group in enumerate(groups_sorted):
            secs = group_to_sections.get(gi, [])
            if not secs:
                continue
            # assign across listed sections in order
            si = 0
            for sku in group:
                # move to next section if current exhausted
                while si < len(secs):
                    skey = secs[si]
                    try:
                        loc = next(sec_iters[skey])
                        assignment[sku] = loc
                        break
                    except StopIteration:
                        si += 1
                if si >= len(secs):
                    break

        return assignment

        # (removed duplicate dead code)
