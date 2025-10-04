from __future__ import annotations
from typing import Dict, Callable

# Thin registry so adding/changing policies stays easy
_POLICIES: Dict[str, str] = {
    "affinity": "AffinitySlotting",
    "abc": "PopularityABCSlotting",
    "rr": "RoundRobinSlotting",
    "rand": "RandomSlotting",
}


def list_policies():
    return sorted(_POLICIES.keys())
