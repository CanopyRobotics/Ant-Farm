from typing import List, Dict
from models import WarehouseCfg, Order

class SimulationEngine:
    def __init__(self, warehouse_cfg, slotting_policy, batching_policy, routing_policy):
        self.warehouse_cfg = warehouse_cfg
        self.slotting_policy = slotting_policy
        self.batching_policy = batching_policy
        self.routing_policy = routing_policy

    def run(self, sku_ids, orders, locations, sections_per_side, num_pickers=1):
        sku_to_location = self.slotting_policy.assign(sku_ids, locations)
        orders_by_picker = self.batching_policy.batch(orders, num_pickers)
        picker_path = self.routing_policy.build_path(
            orders_by_picker, sku_to_location, self.warehouse_cfg, sections_per_side=sections_per_side
        )
        return picker_path