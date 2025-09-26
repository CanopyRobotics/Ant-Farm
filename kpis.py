def compute_kpis(picker_paths, orders_by_picker, wh):
    total_distance = 0.0
    total_time = 0.0
    per_picker_distances = []
    per_picker_times = []
    per_picker_kpis = []
    orders_per_picker = []
    steps_per_picker = []
    max_distance = 0.0
    min_distance = float('inf')
    max_time = 0.0
    min_time = float('inf')
    total_orders = 0
    total_lines = 0

    for idx, picker_path in enumerate(picker_paths):
        batch_orders = orders_by_picker[idx]
        num_orders = len(batch_orders)
        orders_per_picker.append(num_orders)
        steps = len(picker_path)
        steps_per_picker.append(steps)
        lines = sum(len(order.lines) for order in batch_orders)
        total_orders += num_orders
        total_lines += lines

        if not picker_path or len(picker_path) < 2:
            distance = 0.0
            time = 0.0
        else:
            distance = 0.0
            for i in range(1, len(picker_path)):
                x1, y1 = picker_path[i-1]
                x2, y2 = picker_path[i]
                distance += ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5
            time = distance / wh.speed_mps + steps * wh.pick_time_s

        per_picker_distances.append(distance)
        per_picker_times.append(time)
        total_distance += distance
        total_time += time

        max_distance = max(max_distance, distance)
        min_distance = min(min_distance, distance)
        max_time = max(max_time, time)
        min_time = min(min_time, time)

        per_picker_kpis.append({
            "Picker": idx + 1,
            "Orders Completed": num_orders,
            "Distance Walked (m)": distance,
            "Time (s)": time,
            "Time (min)": time / 60 if time else 0,
            "Steps": steps,
            "Lines Picked": lines
        })

    avg_distance = sum(per_picker_distances) / len(per_picker_distances) if per_picker_distances else 0
    avg_time = sum(per_picker_times) / len(per_picker_times) if per_picker_times else 0
    avg_orders = sum(orders_per_picker) / len(orders_per_picker) if orders_per_picker else 0
    avg_lines_per_order = total_lines / total_orders if total_orders else 0

    kpis = {
        "Per Picker": per_picker_kpis,
        "Operation": {
            "Total Distance Walked (m)": total_distance,
            "Total Time (s)": total_time,
            "Total Time (min)": total_time / 60 if total_time else 0,
            "Average Picker Distance (m)": avg_distance,
            "Average Picker Time (s)": avg_time,
            "Average Picker Time (min)": avg_time / 60 if avg_time else 0,
            "Max Picker Distance (m)": max_distance,
            "Min Picker Distance (m)": min_distance,
            "Max Picker Time (s)": max_time,
            "Min Picker Time (s)": min_time,
            "Orders per Picker": orders_per_picker,
            "Average Orders per Picker": avg_orders,
            "Average Lines per Order": avg_lines_per_order,
            "Pickers": len(picker_paths),
            "Total Orders": total_orders,
            "Total Lines Picked": total_lines
        }
    }
    return kpis