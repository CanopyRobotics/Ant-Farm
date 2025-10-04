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
            # Use aisle centerline for distance to I/O to avoid systematic left/right bias
            x_left_rack = (aisle - 1) * (2 * rack_depth + aisle_width)
            x_aisle = x_left_rack + rack_depth  # center walking lane
            x = x_aisle
            s_idx = _section_idx(section)
            y = s_idx * section_length + section_length / 2.0
            return x, y

        def _cost_from_xy(x: float, y: float) -> float:
            # Favor bottom (y) much more than inner aisles (x) so hotspots gravitate to warehouse bottom across all aisles
            wx = 0.2  # aisle spread penalty (small)
            wy = 1.0  # depth penalty (dominant)
            return wx * abs(x) + wy * y

        def _cost(side_id: str, section: str) -> float:
            x, y = _section_center_xy(side_id, section)
            return _cost_from_xy(x, y)

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

        # Order sections by depth bands (section letters) first, then interleave aisles and sides within each band
        # This spreads hotspots along the bottom across all aisles instead of exhausting one aisle at a time.
        def _sec_letters(sec: str) -> str:
            try:
                return str(sec).split('-')[-1]
            except Exception:
                return str(sec)

        from collections import defaultdict
        def _letters_idx(letters: str) -> int:
            try:
                a = ord(letters[0]) - ord('A')
                b = ord(letters[1]) - ord('A') if len(letters) > 1 else 0
                return a * 26 + b
            except Exception:
                return 0

        # Build availability map per (aisle, side, letters)
        sec_lookup: Dict[tuple, tuple] = {}
        aisles_present = set()
        bands_present = set()
        for (sid, sec) in section_buckets.keys():
            aisle = _aisle_from_side(sid)
            side = 'L' if _is_left(sid) else 'R'
            letters = _sec_letters(sec)
            sec_lookup[(aisle, side, letters)] = (sid, sec)
            aisles_present.add(aisle)
            bands_present.add(letters)

        aisles_sorted = sorted(aisles_present)
        bands_sorted = sorted(bands_present, key=_letters_idx)  # bottom to top

        # Build ordered section keys by iterating bands then interleaving aisles/sides inside each band
        sections_ordered: List[tuple] = []
        for bi, letters in enumerate(bands_sorted):
            # Alternate aisle sweep direction per band to avoid drift
            aisles_band = aisles_sorted if (bi % 2 == 0) else list(reversed(aisles_sorted))
            # Alternate which side starts per band
            start_side = 'L' if (bi % 2 == 0) else 'R'
            other_side = 'R' if start_side == 'L' else 'L'
            # Within the band, traverse aisles and add start_side then other_side if present
            for a in aisles_band:
                first = sec_lookup.get((a, start_side, letters))
                second = sec_lookup.get((a, other_side, letters))
                if first:
                    sections_ordered.append(first)
                if second:
                    sections_ordered.append(second)

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

        # Build sections by band for balanced assignment
        sections_by_band: Dict[str, List[tuple]] = {letters: [] for letters in bands_sorted}
        for bi, letters in enumerate(bands_sorted):
            aisles_band = aisles_sorted if (bi % 2 == 0) else list(reversed(aisles_sorted))
            start_side = 'L' if (bi % 2 == 0) else 'R'
            other_side = 'R' if start_side == 'L' else 'L'
            for a in aisles_band:
                first = sec_lookup.get((a, start_side, letters))
                second = sec_lookup.get((a, other_side, letters))
                if first:
                    sections_by_band[letters].append(first)
                if second:
                    sections_by_band[letters].append(second)

        # Greedy balanced placement: take next hottest group and place it into the band slot
        # that minimizes per-aisle cumulative weight after placement.
        group_to_sections: Dict[int, List[tuple]] = {}
        # order groups by weight descending, keep indices
        g_order = list(range(len(groups_sorted)))
        g_order.sort(key=lambda gi: _group_weight(groups_sorted[gi]), reverse=True)
        from collections import defaultdict
        aisle_load = defaultdict(float)  # cumulative assigned weight per aisle
        # helper to get aisle from section key
        def _aisle_of_skey(skey: tuple) -> int:
            return _aisle_from_side(skey[0])

        # iterate bands from bottom to top
        go_idx = 0
        for letters in bands_sorted:
            slots = list(sections_by_band.get(letters, []))
            used = set()
            # fill as many slots in this band as available or until we run out of groups
            while go_idx < len(g_order) and len(used) < len(slots):
                gi = g_order[go_idx]
                w = _group_weight(groups_sorted[gi])
                # pick slot that results in minimal aisle load after placing this group
                best_k = None
                best_cost = None
                for k, skey in enumerate(slots):
                    if k in used:
                        continue
                    a = _aisle_of_skey(skey)
                    cost = aisle_load[a] + w
                    if best_cost is None or cost < best_cost - 1e-12:
                        best_cost = cost
                        best_k = k
                if best_k is None:
                    break
                skey = slots[best_k]
                used.add(best_k)
                a = _aisle_of_skey(skey)
                aisle_load[a] += w
                group_to_sections[gi] = [skey]
                go_idx += 1
                # if group is larger than capacity, spill to next nearest slots in same band first
                remaining = len(groups_sorted[gi]) - len(section_buckets[skey])
                while remaining > 0:
                    # choose next best available slot in same band
                    next_k = None
                    next_cost = None
                    for k, sk in enumerate(slots):
                        if k in used:
                            continue
                        a2 = _aisle_of_skey(sk)
                        cc = aisle_load[a2] + w
                        if next_cost is None or cc < next_cost - 1e-12:
                            next_cost = cc
                            next_k = k
                    if next_k is None:
                        break
                    sk2 = slots[next_k]
                    used.add(next_k)
                    a2 = _aisle_of_skey(sk2)
                    aisle_load[a2] += w
                    group_to_sections[gi].append(sk2)
                    remaining -= len(section_buckets[sk2])
        # Any leftover groups (if bands exhausted) map to remaining sections in overall order
        if len(group_to_sections) < len(groups_sorted):
            remaining_gis = [gi for gi in g_order if gi not in group_to_sections]
            remaining_secs = [s for s in sections_ordered if s not in {ss for lst in group_to_sections.values() for ss in lst}]
            for gi, skey in zip(remaining_gis, remaining_secs):
                group_to_sections[gi] = [skey]

        # Balanced hill-climb: swap assignments to reduce weighted distance and per-aisle L/R heat imbalance
        # Applies when groups map 1:1 to sections (group_size ≈ section_capacity)
        if all(len(g) >= 1 for g in groups_sorted):
            # Candidate assignments: (group_index, section_key)
            single_assign = [(gi, secs[0]) for gi, secs in group_to_sections.items() if len(secs) == 1]
            if single_assign:
                # Precompute per-section geometry and normalized distances
                sec_dist = {skey: _cost(skey[0], skey[1]) for skey in sections_ordered}
                max_d = max(sec_dist.values()) if sec_dist else 1.0
                sec_dnorm = {k: (v / max_d if max_d > 0 else 0.0) for k, v in sec_dist.items()}

                # Group weights (popularity) and normalized weights
                gw = {gi: _group_weight(groups_sorted[gi]) for gi, _ in single_assign}
                mean_w = (sum(gw.values()) / len(gw)) if gw else 1.0
                if mean_w <= 0:
                    mean_w = 1.0
                gw_norm = {gi: (w / mean_w) for gi, w in gw.items()}

                # Build per-aisle left/right weighted sums
                def _side_char(sid: str) -> str:
                    return 'L' if _is_left(sid) else 'R'

                from collections import defaultdict
                aisle_w = defaultdict(lambda: {'L': 0.0, 'R': 0.0})
                for gi, skey in single_assign:
                    sid = skey[0]
                    a = _aisle_from_side(sid)
                    sc = _side_char(sid)
                    aisle_w[a][sc] += gw_norm.get(gi, 1.0)

                def imbalance_cost() -> float:
                    return sum(abs(v['L'] - v['R']) for v in aisle_w.values())

                lam = 0.15  # weight for imbalance vs distance (tunable)

                improved = True
                iters = 0
                # Make a local map for quick index lookup
                idx_of = {gi: i for i, (gi, _) in enumerate(single_assign)}
                while improved and iters < 300:
                    improved = False
                    iters += 1
                    # Greedy pass: try all pairs; break on first improvement
                    for i in range(len(single_assign)):
                        gi1, s1 = single_assign[i]
                        sid1, _sec1 = s1
                        a1 = _aisle_from_side(sid1)
                        sc1 = _side_char(sid1)
                        for j in range(i+1, len(single_assign)):
                            gi2, s2 = single_assign[j]
                            sid2, _sec2 = s2
                            a2 = _aisle_from_side(sid2)
                            sc2 = _side_char(sid2)
                            # Distance delta (normalized weights and distances)
                            d_cur = gw_norm[gi1]*sec_dnorm[s1] + gw_norm[gi2]*sec_dnorm[s2]
                            d_alt = gw_norm[gi1]*sec_dnorm[s2] + gw_norm[gi2]*sec_dnorm[s1]
                            dd = d_alt - d_cur
                            # Imbalance delta: adjust only affected aisles
                            old_imb = 0.0
                            new_imb = 0.0
                            # Aisle a1
                            L1, R1 = aisle_w[a1]['L'], aisle_w[a1]['R']
                            if sc1 == 'L':
                                L1_old = L1; L1_new = L1 - gw_norm[gi1] + (gw_norm[gi2] if a1 == a2 and sc2 == 'L' else 0.0)
                                R1_old = R1; R1_new = R1 + (gw_norm[gi2] if a1 == a2 and sc2 == 'R' else 0.0)
                            else:
                                R1_old = R1; R1_new = R1 - gw_norm[gi1] + (gw_norm[gi2] if a1 == a2 and sc2 == 'R' else 0.0)
                                L1_old = L1; L1_new = L1 + (gw_norm[gi2] if a1 == a2 and sc2 == 'L' else 0.0)
                            old_imb += abs(L1 - R1)
                            new_imb += abs(L1_new - R1_new)
                            # Aisle a2 (if different)
                            if a2 != a1:
                                L2, R2 = aisle_w[a2]['L'], aisle_w[a2]['R']
                                if sc2 == 'L':
                                    L2_old = L2; L2_new = L2 - gw_norm[gi2] + gw_norm[gi1]
                                    R2_old = R2; R2_new = R2
                                else:
                                    R2_old = R2; R2_new = R2 - gw_norm[gi2] + gw_norm[gi1]
                                    L2_old = L2; L2_new = L2
                                old_imb += abs(L2 - R2)
                                new_imb += abs(L2_new - R2_new)
                            dimb = new_imb - old_imb

                            if dd + lam * dimb < -1e-9:
                                # accept swap
                                single_assign[i] = (gi1, s2)
                                single_assign[j] = (gi2, s1)
                                group_to_sections[gi1] = [s2]
                                group_to_sections[gi2] = [s1]
                                # update aisle weights
                                if a1 == a2:
                                    if sc1 == 'L':
                                        aisle_w[a1]['L'] += -gw_norm[gi1] + gw_norm[gi2]
                                    else:
                                        aisle_w[a1]['R'] += -gw_norm[gi1] + gw_norm[gi2]
                                    if sc2 == 'L':
                                        aisle_w[a1]['L'] += -gw_norm[gi2] + gw_norm[gi1]
                                    else:
                                        aisle_w[a1]['R'] += -gw_norm[gi2] + gw_norm[gi1]
                                else:
                                    # remove old
                                    aisle_w[a1][sc1] -= gw_norm[gi1]
                                    aisle_w[a2][sc2] -= gw_norm[gi2]
                                    # add new
                                    aisle_w[a1][sc2] += gw_norm[gi2]
                                    aisle_w[a2][sc1] += gw_norm[gi1]
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
