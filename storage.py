from enum import Enum
from dataclasses import dataclass
from typing import List, Dict

class AisleSide(Enum):
    LEFT = "L"
    RIGHT = "R"

@dataclass(frozen=True)
class StorageLocation:
    side_id: str
    section: str
    tote_id: str

    @property
    def aisle(self) -> int:
        side_num = int(self.side_id[1:])
        return (side_num + 1) // 2

def gen_storage_locations(num_aisles: int, sections_per_side: int) -> List[StorageLocation]:
    locations = []
    for aisle in range(1, num_aisles + 1):
        left_side_id = f"A{2*aisle-1:03d}"
        right_side_id = f"A{2*aisle:03d}"
        for side_id in [left_side_id, right_side_id]:
            for s_idx in range(sections_per_side):
                section_letters = chr(ord('A') + (s_idx // 26)) + chr(ord('A') + (s_idx % 26))
                section_id = f"{side_id}-{section_letters}"
                for level in range(1, 7):
                    for tote in range(1, 4):
                        tote_num = (level - 1) * 10 + tote
                        tote_id = f"{section_id}-{tote_num:03d}"
                        locations.append(StorageLocation(
                            side_id=side_id,
                            section=section_id,
                            tote_id=tote_id
                        ))
    return locations

def assign_skus_to_locations(sku_ids: List[int], locations: List[StorageLocation]) -> Dict[int, StorageLocation]:
    mapping = {}
    n = len(locations)
    for idx, sku in enumerate(sku_ids):
        mapping[sku] = locations[idx % n]
    return mapping