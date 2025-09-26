from collections import defaultdict

class RoutingPolicy:
    def build_path(self, orders_by_picker, sku_to_location, warehouse_cfg, **kwargs):
        raise NotImplementedError

class SideGroupedRouting(RoutingPolicy):
    def build_path(self, orders_by_picker, sku_to_location, warehouse_cfg, **kwargs):
        # This is your build_side_grouped_picker_path logic
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
        return build_side_grouped_picker_path(pick_points, start_point, side_xs, total_height, cross_aisle_height)

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

import random

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