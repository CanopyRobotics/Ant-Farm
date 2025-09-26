from typing import List, Dict
from models import WarehouseCfg, Order

class SimulationEngine:
    def __init__(self, wh, slotting_policy, batching_policy, routing_policy):
        self.wh = wh
        self.slotting_policy = slotting_policy
        self.batching_policy = batching_policy
        self.routing_policy = routing_policy

    def run(self, sku_ids, orders, locations, sections_per_side, num_pickers=1):
        sku_to_location = self.slotting_policy.assign(sku_ids, locations)
        orders_by_picker = self.batching_policy.batch(orders, num_pickers)
        picker_paths = self.routing_policy.build_path(
            orders_by_picker, sku_to_location, self.wh, sections_per_side=sections_per_side
        )
        return picker_paths

def location_to_xy(loc, wh, sections_per_side):
    """
    Convert a StorageLocation to (x, y) coordinates for plotting/pathing.
    """
    # Parse aisle index
    aisle_idx = int(loc.side_id[1:]) // 2  # 0-based
    is_left = int(loc.side_id[1:]) % 2 == 1
    rack_depth = 0.5
    aisle_width = wh.aisle_width_m
    section_length = wh.aisle_length_m / sections_per_side

    # Compute x position
    x_left_rack = aisle_idx * (2 * rack_depth + aisle_width)
    x_aisle = x_left_rack + rack_depth
    x_right_rack = x_aisle + aisle_width
    x = x_left_rack + rack_depth / 2 if is_left else x_right_rack + rack_depth / 2

    # Compute section index from section letters
    section_letters = loc.section.split('-')[-1]
    s_idx = (ord(section_letters[0]) - ord('A')) * 26 + (ord(section_letters[1]) - ord('A'))
    y = s_idx * section_length + section_length / 2

    return x, y