from collections import defaultdict
import random

class RoutingPolicy:
    def build_path(self, orders_by_picker, sku_to_location, warehouse_cfg, **kwargs):
        raise NotImplementedError

    @staticmethod
    def warehouse_distance(p1, p2, wh, sections_per_side):
        """
        Compute shortest valid warehouse walking distance between two points.
        Only allows movement along aisles and cross-aisles.
        """
        aisle_width = wh.aisle_width_m
        rack_depth = 0.5
        cross_aisle_height = 2.0
        aisle_length = wh.aisle_length_m

        # Cross-aisles at y = -cross_aisle_height/2 and y = aisle_length + cross_aisle_height/2
        y_bot = -cross_aisle_height / 2
        y_top = aisle_length + cross_aisle_height / 2

        y1, y2 = p1[1], p2[1]
        entry1 = y_bot if abs(y1 - y_bot) < abs(y1 - y_top) else y_top
        entry2 = y_bot if abs(y2 - y_bot) < abs(y2 - y_top) else y_top

        # Walk: vertical to cross-aisle, horizontal to target aisle, vertical to pick
        dist = abs(y1 - entry1) + abs(p1[0] - p2[0]) + abs(y2 - entry2)
        if entry1 != entry2:
            dist += abs(entry1 - entry2)
        return dist

class SideGroupedRouting(RoutingPolicy):
    def build_path(self, orders_by_picker, sku_to_location, warehouse_cfg, **kwargs):
        wh = warehouse_cfg
        sections_per_side = kwargs.get("sections_per_side", 10)
        aisle_width = wh.aisle_width_m
        aisle_length = wh.aisle_length_m
        rack_depth = 0.5
        section_length = aisle_length / sections_per_side
        cross_aisle_height = 2.0

        num_aisles = wh.num_aisles
        x_positions = []
        for aisle in range(num_aisles):
            x_left_rack = aisle * (2 * rack_depth + aisle_width)
            x_aisle = x_left_rack + rack_depth
            x_right_rack = x_aisle + aisle_width
            x_positions.append((x_left_rack, x_aisle, x_right_rack))
        total_height = sections_per_side * section_length

        all_picker_paths = []
        for picker_orders in orders_by_picker:
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
            all_picker_paths.append(picker_path)
        return all_picker_paths

def build_side_grouped_picker_path(pick_points, start_point, side_xs, total_height, cross_aisle_height):
    side_picks = defaultdict(list)
    for x, y, side_id in pick_points:
        side_picks[side_id].append((x, y))
    sorted_sides = sorted(side_picks.keys(), key=lambda s: int(s[1:]))
    y_bot = -cross_aisle_height / 2
    y_top = total_height + cross_aisle_height / 2
    picker_path = [start_point]
    current_pos = start_point
    for side_id in sorted_sides:
        picks = side_picks[side_id]
        picks_sorted = sorted(picks, key=lambda p: p[1])
        x_side = side_xs[side_id]
        dist_to_bot = abs(current_pos[1] - y_bot)
        dist_to_top = abs(current_pos[1] - y_top)
        entry_y = y_bot if dist_to_bot <= dist_to_top else y_top
        if current_pos[1] != entry_y:
            picker_path.append((current_pos[0], entry_y))
        if current_pos[0] != x_side:
            picker_path.append((x_side, entry_y))
        picks_in_order = sorted(picks, key=lambda p: p[1]) if entry_y == y_bot else sorted(picks, key=lambda p: -p[1])
        for x, y in picks_in_order:
            if picker_path[-1] != (x, y):
                picker_path.append((x, y))
        current_pos = picker_path[-1]
    return picker_path

class RandomRouting(RoutingPolicy):
    def build_path(self, orders_by_picker, sku_to_location, warehouse_cfg, **kwargs):
        wh = warehouse_cfg
        sections_per_side = kwargs.get("sections_per_side", 10)
        aisle_width = wh.aisle_width_m
        aisle_length = wh.aisle_length_m
        rack_depth = 0.5
        section_length = aisle_length / sections_per_side
        cross_aisle_height = 2.0

        num_aisles = wh.num_aisles
        x_positions = []
        for aisle in range(num_aisles):
            x_left_rack = aisle * (2 * rack_depth + aisle_width)
            x_aisle = x_left_rack + rack_depth
            x_right_rack = x_aisle + aisle_width
            x_positions.append((x_left_rack, x_aisle, x_right_rack))

        picker_orders = orders_by_picker[0]
        pick_points = []
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
                pick_points.append((x, y))
        rng = random.Random()
        rng.shuffle(pick_points)
        start_point = (x_positions[0][1] + aisle_width / 2, -cross_aisle_height / 2)
        return [start_point] + pick_points

class SShapeRouting(RoutingPolicy):
    def build_path(self, orders_by_picker, sku_to_location, warehouse_cfg, **kwargs):
        wh = warehouse_cfg
        sections_per_side = kwargs.get("sections_per_side", 10)
        aisle_width = wh.aisle_width_m
        aisle_length = wh.aisle_length_m
        rack_depth = 0.5
        section_length = aisle_length / sections_per_side
        cross_aisle_height = 2.0

        num_aisles = wh.num_aisles
        x_positions = []
        for aisle in range(num_aisles):
            x_left_rack = aisle * (2 * rack_depth + aisle_width)
            x_aisle = x_left_rack + rack_depth
            x_right_rack = x_aisle + aisle_width
            x_positions.append((x_left_rack, x_aisle, x_right_rack))
        total_height = sections_per_side * section_length

        all_picker_paths = []
        for picker_orders in orders_by_picker:
            aisle_side_picks = defaultdict(list)
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
                    aisle_side_picks[(aisle_idx, is_left)].append((x, y))

            sorted_sides = sorted(
                aisle_side_picks.keys(),
                key=lambda k: (k[0], 0 if k[1] else 1)
            )

            # Start at bottom cross-aisle, center of first aisle
            start_point = (x_positions[0][1] + aisle_width / 2, -cross_aisle_height / 2)
            path = [start_point]
            current_x, current_y = start_point

            for idx, (aisle_idx, is_left) in enumerate(sorted_sides):
                picks = aisle_side_picks[(aisle_idx, is_left)]
                if not picks:
                    continue
                x_side = x_positions[aisle_idx][0] + rack_depth / 2 if is_left else x_positions[aisle_idx][2] + rack_depth / 2

                # Decide entry and exit cross-aisle for this aisle side
                # If first aisle side, always enter from bottom cross-aisle
                if idx == 0:
                    entry_y = -cross_aisle_height / 2
                    exit_y = total_height + cross_aisle_height / 2
                else:
                    # Use whichever cross-aisle the picker is currently at
                    entry_y = current_y
                    exit_y = total_height + cross_aisle_height / 2 if entry_y == -cross_aisle_height / 2 else -cross_aisle_height / 2

                # Move horizontally in current cross-aisle to aisle entry
                if (current_x, current_y) != (x_side, entry_y):
                    if current_x != x_side:
                        path.append((x_side, current_y))
                    if current_y != entry_y:
                        path.append((x_side, entry_y))

                # Traverse full aisle from entry to exit, picking everything
                picks_sorted = sorted(picks, key=lambda p: p[1]) if entry_y < exit_y else sorted(picks, key=lambda p: -p[1])
                for x, y in picks_sorted:
                    if path[-1] != (x, y):
                        path.append((x, y))
                # Go to exit cross-aisle at end of aisle
                if path[-1] != (x_side, exit_y):
                    path.append((x_side, exit_y))
                current_x, current_y = x_side, exit_y

            # Picker ends at the last cross-aisle reached
            all_picker_paths.append(path)
        return all_picker_paths

class LargestGapRouting(RoutingPolicy):
    def build_path(self, orders_by_picker, sku_to_location, warehouse_cfg, **kwargs):
        wh = warehouse_cfg
        sections_per_side = kwargs.get("sections_per_side", 10)
        aisle_width = wh.aisle_width_m
        aisle_length = wh.aisle_length_m
        rack_depth = 0.5
        section_length = aisle_length / sections_per_side
        cross_aisle_height = 2.0

        num_aisles = wh.num_aisles
        x_positions = []
        for aisle in range(num_aisles):
            x_left_rack = aisle * (2 * rack_depth + aisle_width)
            x_aisle = x_left_rack + rack_depth
            x_right_rack = x_aisle + aisle_width
            x_positions.append((x_left_rack, x_aisle, x_right_rack))
        total_height = sections_per_side * section_length

        all_picker_paths = []
        for picker_orders in orders_by_picker:
            aisle_side_picks = defaultdict(list)
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
                    aisle_side_picks[(aisle_idx, is_left)].append((x, y))

            sorted_sides = sorted(
                aisle_side_picks.keys(),
                key=lambda k: (k[0], 0 if k[1] else 1)
            )

            # Start at bottom cross-aisle, center of first aisle
            start_point = (x_positions[0][1] + aisle_width / 2, -cross_aisle_height / 2)
            path = [start_point]
            current_x, current_y = start_point

            for (aisle_idx, is_left) in sorted_sides:
                picks = aisle_side_picks[(aisle_idx, is_left)]
                if not picks:
                    continue
                x_side = x_positions[aisle_idx][0] + rack_depth / 2 if is_left else x_positions[aisle_idx][2] + rack_depth / 2

                # Choose entry cross-aisle (nearest to current position)
                y_bot = -cross_aisle_height / 2
                y_top = total_height + cross_aisle_height / 2
                dist_to_bot = abs(current_y - y_bot)
                dist_to_top = abs(current_y - y_top)
                entry_y = y_bot if dist_to_bot <= dist_to_top else y_top

                # Move horizontally in cross-aisle to aisle entry
                if (current_x, current_y) != (x_side, entry_y):
                    if current_y != entry_y:
                        path.append((current_x, entry_y))
                    if current_x != x_side:
                        path.append((x_side, entry_y))

                # Find largest gap among picks (including entry and exit points)
                picks_sorted = sorted(picks, key=lambda p: p[1])
                positions = [entry_y] + [y for x, y in picks_sorted] + [y_top if entry_y == y_bot else y_bot]
                gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                largest_gap_idx = int(max(range(len(gaps)), key=lambda i: abs(gaps[i])))

                # If largest gap is at the start, enter from the opposite cross-aisle
                if largest_gap_idx == 0:
                    entry_y = y_top if entry_y == y_bot else y_bot
                    # Move horizontally in cross-aisle to aisle entry
                    if (current_x, current_y) != (x_side, entry_y):
                        if current_y != entry_y:
                            path.append((current_x, entry_y))
                        if current_x != x_side:
                            path.append((x_side, entry_y))
                    # Walk to all picks
                    for x, y in picks_sorted:
                        if path[-1] != (x, y):
                            path.append((x, y))
                    # Exit at far cross-aisle
                    if path[-1] != (x_side, positions[-1]):
                        path.append((x_side, positions[-1]))
                    current_x, current_y = x_side, positions[-1]
                # If largest gap is at the end, walk all the way to last pick and exit at far cross-aisle
                elif largest_gap_idx == len(gaps) - 1:
                    for x, y in picks_sorted:
                        if path[-1] != (x, y):
                            path.append((x, y))
                    # Exit at far cross-aisle
                    if path[-1] != (x_side, positions[-1]):
                        path.append((x_side, positions[-1]))
                    current_x, current_y = x_side, positions[-1]
                # Otherwise, walk to last pick before largest gap, turn back, and exit at entry
                else:
                    walk_picks = picks_sorted[:largest_gap_idx]
                    for x, y in walk_picks:
                        if path[-1] != (x, y):
                            path.append((x, y))
                    # Turn back at largest gap, return to entry
                    if path[-1] != (x_side, entry_y):
                        path.append((x_side, entry_y))
                    current_x, current_y = x_side, entry_y

            all_picker_paths.append(path)
        return all_picker_paths

class HybridCombinedRouting(RoutingPolicy):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def build_path(self, orders_by_picker, sku_to_location, warehouse_cfg, **kwargs):
        wh = warehouse_cfg
        sections_per_side = kwargs.get("sections_per_side", 10)
        aisle_width = wh.aisle_width_m
        aisle_length = wh.aisle_length_m
        rack_depth = 0.5
        section_length = aisle_length / sections_per_side
        cross_aisle_height = 2.0

        num_aisles = wh.num_aisles
        x_positions = []
        for aisle in range(num_aisles):
            x_left_rack = aisle * (2 * rack_depth + aisle_width)
            x_aisle = x_left_rack + rack_depth
            x_right_rack = x_aisle + aisle_width
            x_positions.append((x_left_rack, x_aisle, x_right_rack))
        total_height = sections_per_side * section_length

        all_picker_paths = []
        for picker_orders in orders_by_picker:
            aisle_side_picks = defaultdict(list)
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
                    aisle_side_picks[(aisle_idx, is_left)].append((x, y))

            sorted_sides = sorted(
                aisle_side_picks.keys(),
                key=lambda k: (k[0], 0 if k[1] else 1)
            )

            start_point = (x_positions[0][1] + aisle_width / 2, -cross_aisle_height / 2)
            path = [start_point]
            current_x, current_y = start_point

            for (aisle_idx, is_left) in sorted_sides:
                picks = aisle_side_picks[(aisle_idx, is_left)]
                if not picks:
                    continue
                x_side = x_positions[aisle_idx][0] + rack_depth / 2 if is_left else x_positions[aisle_idx][2] + rack_depth / 2

                # Choose entry cross-aisle (nearest to current position)
                y_bot = -cross_aisle_height / 2
                y_top = total_height + cross_aisle_height / 2
                dist_to_bot = abs(current_y - y_bot)
                dist_to_top = abs(current_y - y_top)
                entry_y = y_bot if dist_to_bot <= dist_to_top else y_top

                # Move horizontally in cross-aisle to aisle entry
                if (current_x, current_y) != (x_side, entry_y):
                    if current_y != entry_y:
                        path.append((current_x, entry_y))
                    if current_x != x_side:
                        path.append((x_side, entry_y))

                # --- Hybrid logic ---
                picks_sorted = sorted(picks, key=lambda p: p[1])
                if len(picks_sorted) == 0:
                    continue  # No picks, skip aisle side

                if len(picks_sorted) <= self.threshold:
                    # Largest Gap logic
                    positions = [entry_y] + [y for x, y in picks_sorted] + [y_top if entry_y == y_bot else y_bot]
                    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                    largest_gap_idx = int(max(range(len(gaps)), key=lambda i: abs(gaps[i])))

                    # Determine picks to walk before turning
                    if largest_gap_idx == 0 or largest_gap_idx == len(gaps) - 1:
                        # If largest gap is at the start or end, only walk if there are picks between entry and exit
                        if len(picks_sorted) > 0:
                            for x, y in picks_sorted:
                                if path[-1] != (x, y):
                                    path.append((x, y))
                            exit_y = positions[-1]
                            if path[-1] != (x_side, exit_y):
                                path.append((x_side, exit_y))
                            current_x, current_y = x_side, exit_y
                        else:
                            continue  # No picks between entry and exit, skip aisle side
                    else:
                        walk_picks = picks_sorted[:largest_gap_idx]
                        if len(walk_picks) == 0:
                            continue  # No picks before the gap, skip aisle side
                        for x, y in walk_picks:
                            if path[-1] != (x, y):
                                path.append((x, y))
                        if path[-1] != (x_side, entry_y):
                            path.append((x_side, entry_y))
                        current_x, current_y = x_side, entry_y
                else:
                    # S-Shape logic
                    exit_y = y_top if entry_y == y_bot else y_bot
                    picks_ordered = sorted(picks, key=lambda p: p[1]) if entry_y < exit_y else sorted(picks, key=lambda p: -p[1])
                    for x, y in picks_ordered:
                        if path[-1] != (x, y):
                            path.append((x, y))
                    if path[-1] != (x_side, exit_y):
                        path.append((x_side, exit_y))
                    current_x, current_y = x_side, exit_y

            all_picker_paths.append(path)
        return all_picker_paths

class PenetrationThresholdRouting(RoutingPolicy):
    def __init__(self, threshold=0.5):
        self.threshold = threshold  # Fraction of aisle length (e.g., 0.5 for 50%)

    def build_path(self, orders_by_picker, sku_to_location, warehouse_cfg, **kwargs):
        wh = warehouse_cfg
        sections_per_side = kwargs.get("sections_per_side", 10)
        aisle_width = wh.aisle_width_m
        aisle_length = wh.aisle_length_m
        rack_depth = 0.5
        section_length = aisle_length / sections_per_side
        cross_aisle_height = 2.0

        num_aisles = wh.num_aisles
        x_positions = []
        for aisle in range(num_aisles):
            x_left_rack = aisle * (2 * rack_depth + aisle_width)
            x_aisle = x_left_rack + rack_depth
            x_right_rack = x_aisle + aisle_width
            x_positions.append((x_left_rack, x_aisle, x_right_rack))
        total_height = sections_per_side * section_length

        all_picker_paths = []
        for picker_orders in orders_by_picker:
            aisle_side_picks = defaultdict(list)
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
                    aisle_side_picks[(aisle_idx, is_left)].append((x, y))

            sorted_sides = sorted(
                aisle_side_picks.keys(),
                key=lambda k: (k[0], 0 if k[1] else 1)
            )

            start_point = (x_positions[0][1] + aisle_width / 2, -cross_aisle_height / 2)
            path = [start_point]
            current_x, current_y = start_point

            for (aisle_idx, is_left) in sorted_sides:
                picks = aisle_side_picks[(aisle_idx, is_left)]
                if not picks:
                    continue
                x_side = x_positions[aisle_idx][0] + rack_depth / 2 if is_left else x_positions[aisle_idx][2] + rack_depth / 2

                # Choose entry cross-aisle (nearest to current position)
                y_bot = -cross_aisle_height / 2
                y_top = total_height + cross_aisle_height / 2
                dist_to_bot = abs(current_y - y_bot)
                dist_to_top = abs(current_y - y_top)
                entry_y = y_bot if dist_to_bot <= dist_to_top else y_top

                # Move horizontally in cross-aisle to aisle entry
                if (current_x, current_y) != (x_side, entry_y):
                    if current_y != entry_y:
                        path.append((current_x, entry_y))
                    if current_x != x_side:
                        path.append((x_side, entry_y))

                # Penetration threshold logic
                picks_sorted = sorted(picks, key=lambda p: p[1])
                # Compute deepest pick (farthest from entry cross-aisle)
                if entry_y == y_bot:
                    deepest_pick = max([y for x, y in picks_sorted])
                    penetration = deepest_pick / aisle_length
                else:
                    deepest_pick = min([y for x, y in picks_sorted])
                    penetration = (aisle_length - deepest_pick) / aisle_length

                if penetration > self.threshold:
                    # S-Shape: traverse full aisle side
                    exit_y = y_top if entry_y == y_bot else y_bot
                    picks_ordered = sorted(picks, key=lambda p: p[1]) if entry_y < exit_y else sorted(picks, key=lambda p: -p[1])
                    for x, y in picks_ordered:
                        if path[-1] != (x, y):
                            path.append((x, y))
                    if path[-1] != (x_side, exit_y):
                        path.append((x_side, exit_y))
                    current_x, current_y = x_side, exit_y
                else:
                    # Largest Gap logic
                    positions = [entry_y] + [y for x, y in picks_sorted] + [y_top if entry_y == y_bot else y_bot]
                    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                    largest_gap_idx = int(max(range(len(gaps)), key=lambda i: abs(gaps[i])))

                    if largest_gap_idx == 0:
                        entry_y = y_top if entry_y == y_bot else y_bot
                        if (current_x, current_y) != (x_side, entry_y):
                            if current_y != entry_y:
                                path.append((current_x, entry_y))
                            if current_x != x_side:
                                path.append((x_side, entry_y))
                    for x, y in picks_sorted:
                        if path[-1] != (x, y):
                            path.append((x, y))
                    # Exit at far cross-aisle
                    if path[-1] != (x_side, positions[-1]):
                        path.append((x_side, positions[-1]))
                    current_x, current_y = x_side, positions[-1]
            all_picker_paths.append(path)
        return all_picker_paths
