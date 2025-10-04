import pandas as pd
from models import Order, OrderLine, WarehouseCfg
from storage import gen_storage_locations
from affinity import compute_affinity_matrix, group_skus_by_affinity
from slotting import AffinitySlotting
from routing import SideGroupedRouting


def _sections_per_side_from_layout(locations):
    secs = {}
    for loc in locations:
        secs.setdefault(loc.side_id, set()).add(loc.section)
    # use median
    counts = [len(v) for v in secs.values()]
    return int(sorted(counts)[len(counts)//2]) if counts else 10


def _orders_from_groups(groups, n_orders=200, mean_lines=4):
    import random
    rng = random.Random(0)
    orders = []
    for i in range(n_orders):
        g = groups[rng.randrange(len(groups))]
        k = max(1, int(rng.expovariate(1.0/mean_lines)))
        chosen = [rng.choice(g) for _ in range(k)]
        orders.append(Order(id=i+1, lines=[OrderLine(sku_id=s) for s in chosen]))
    return orders


def test_affinity_section_contiguity():
    wh = WarehouseCfg(num_aisles=6, aisle_length_m=20.0, aisle_width_m=3.0, speed_mps=0.75, pick_time_s=20.0)
    sections_per_side = 12
    locations = gen_storage_locations(wh.num_aisles, sections_per_side)
    sku_ids = list(range(1, len(locations)+1))

    # synthetic co-pick groups
    groups = [sku_ids[i:i+18] for i in range(0, len(sku_ids), 18)]
    policy = AffinitySlotting(groups)
    mapping = policy.assign(sku_ids, locations)

    # 1) % lines in same section (cluster contiguity proxy)
    orders = _orders_from_groups(groups)
    from collections import Counter
    same_section = 0
    total_lines = 0
    for o in orders:
        secs = [mapping[ln.sku_id].section for ln in o.lines]
        total_lines += len(secs)
        c = Counter(secs)
        same_section += max(c.values())
    pct_same = same_section / max(1, total_lines)
    assert pct_same > 0.5

    # 2) Average intra-order distance should be bounded
    from routing import RoutingPolicy
    def dist(a, b):
        # cheap proxy using aisle distance only
        return abs(mapping[a].aisle - mapping[b].aisle)
    dsum = 0; cnt = 0
    for o in orders:
        ids = [ln.sku_id for ln in o.lines]
        for i in range(len(ids)-1):
            dsum += dist(ids[i], ids[i+1]); cnt += 1
    avg_d = dsum/max(1,cnt)
    assert avg_d < 3

