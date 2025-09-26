from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class OrderLine:
    sku_id: int
    qty: int = 1

@dataclass(frozen=True)
class Order:
    id: int
    lines: List[OrderLine]

@dataclass
class WarehouseCfg:
    num_aisles: int
    aisle_length_m: float
    aisle_width_m: float
    speed_mps: float
    pick_time_s: float