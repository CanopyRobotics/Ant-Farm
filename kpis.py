def compute_kpis(picker_path, orders, wh):
    total_distance = 0.0
    for i in range(1, len(picker_path)):
        x1, y1 = picker_path[i-1]
        x2, y2 = picker_path[i]
        total_distance += ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

    num_picks = sum(len(order.lines) for order in orders)
    num_orders = len(orders)
    walk_time = total_distance / wh.speed_mps  # seconds
    pick_time = num_picks * wh.pick_time_s     # seconds
    total_time = walk_time + pick_time

    avg_distance_per_pick = total_distance / num_picks if num_picks else 0
    avg_picks_per_order = num_picks / num_orders if num_orders else 0
    avg_time_per_order = total_time / num_orders if num_orders else 0
    avg_time_per_pick = total_time / num_picks if num_picks else 0

    return {
        "Total Distance (m)": total_distance,
        "Number of Picks": num_picks,
        "Total Orders": num_orders,
        "Avg Distance per Pick (m)": avg_distance_per_pick,
        "Avg Picks per Order": avg_picks_per_order,
        "Walk Time (s)": walk_time,
        "Pick Time (s)": pick_time,
        "Total Time (s)": total_time,
        "Avg Time per Order (s)": avg_time_per_order,
        "Avg Time per Pick (s)": avg_time_per_pick
    }