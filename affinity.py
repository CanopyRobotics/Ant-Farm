from collections import defaultdict

def compute_affinity_matrix(orders):
    """
    Returns a dict-of-dicts: affinity[sku1][sku2] = co-pick count
    """
    affinity = defaultdict(lambda: defaultdict(int))
    for order in orders:
        sku_list = [line.sku_id for line in order.lines]
        for i in range(len(sku_list)):
            for j in range(i+1, len(sku_list)):
                affinity[sku_list[i]][sku_list[j]] += 1
                affinity[sku_list[j]][sku_list[i]] += 1
    return affinity

def group_skus_by_affinity(affinity, sku_ids, group_size=18):
    """
    Greedily groups SKUs with highest affinity.
    Returns a list of SKU groups (each group is a list of SKUs).
    """
    unassigned = set(sku_ids)
    groups = []
    while unassigned:
        sku = unassigned.pop()
        group = [sku]
        # Find the next most-affined SKUs
        for _ in range(group_size - 1):
            if not unassigned:
                break
            # Pick SKU with highest total affinity to current group
            next_sku = max(
                unassigned,
                key=lambda s: sum(affinity[sku_in_group].get(s, 0) for sku_in_group in group)
            )
            group.append(next_sku)
            unassigned.remove(next_sku)
        groups.append(group)
    return groups