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

        # --- Helper structures for advanced metrics ---
        sku_to_picker = {}  # SKU -> set of pickers
        aisle_visits = []   # List of sets of aisles visited per picker
        penetration_depths = []  # List of max penetration per aisle per picker
        picks_by_class = {'A': 0, 'B': 0, 'C': 0}
        sku_pick_counts = {}  # SKU -> total picks

        # For ABC classification, assume A/B/C sets are available globally (or pass as args)
        try:
            from main import a_skus, b_skus, c_skus
        except ImportError:
            a_skus, b_skus, c_skus = set(), set(), set()

        for idx, picker_path in enumerate(picker_paths):
            batch_orders = orders_by_picker[idx]
            num_orders = len(batch_orders)
            orders_per_picker.append(num_orders)
            steps = len(picker_path)
            steps_per_picker.append(steps)
            lines = sum(len(order.lines) for order in batch_orders)
            total_orders += num_orders
            total_lines += lines

            # --- Distance and time ---
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

            # --- Advanced metrics ---
            # SKU overlap: track which picker picks which SKU
            picker_skus = set()
            for order in batch_orders:
                for line in order.lines:
                    sku = line.sku_id
                    picker_skus.add(sku)
                    sku_pick_counts[sku] = sku_pick_counts.get(sku, 0) + 1
                    if sku in a_skus:
                        picks_by_class['A'] += 1
                    elif sku in b_skus:
                        picks_by_class['B'] += 1
                    elif sku in c_skus:
                        picks_by_class['C'] += 1
                    sku_to_picker.setdefault(sku, set()).add(idx)
            # Aisle utilization: count unique aisles visited
            aisles_visited = set()
            max_penetration = {}  # aisle -> max y
            for point in picker_path:
                x, y = point
                # Estimate aisle index from x (assuming standard layout)
                aisle_idx = int(x // (2 * 0.5 + wh.aisle_width_m))
                aisles_visited.add(aisle_idx)
                max_penetration[aisle_idx] = max(max_penetration.get(aisle_idx, 0), y)
            aisle_visits.append(aisles_visited)
            penetration_depths.append(max_penetration)

            per_picker_kpis.append({
                "Picker": idx + 1,
                "Orders Completed": num_orders,
                "Distance Walked (m)": distance,
                "Time (s)": time,
                "Time (min)": time / 60 if time else 0,
                "Steps": steps,
                "Lines Picked": lines,
                "Unique SKUs Picked": len(picker_skus),
                "Aisles Visited": len(aisles_visited),
                "Max Penetration Depths": max_penetration
            })

    # Old return block removed; new metrics are returned below

        avg_distance = sum(per_picker_distances) / len(per_picker_distances) if per_picker_distances else 0
        avg_time = sum(per_picker_times) / len(per_picker_times) if per_picker_times else 0
        avg_orders = sum(orders_per_picker) / len(orders_per_picker) if orders_per_picker else 0
        avg_lines_per_order = total_lines / total_orders if total_orders else 0
        avg_batch_size = avg_orders  # orders per picker

        # SKU overlap: how many SKUs are picked by >1 picker
        sku_overlap = sum(1 for pickers in sku_to_picker.values() if len(pickers) > 1)

        # Aisle utilization: average aisles visited per picker
        avg_aisles_visited = sum(len(a) for a in aisle_visits) / len(aisle_visits) if aisle_visits else 0

        # Penetration depth: average max penetration per aisle per picker
        all_penetrations = [max_penetration for picker in penetration_depths for max_penetration in picker.values()]
        avg_penetration_depth = sum(all_penetrations) / len(all_penetrations) if all_penetrations else 0

        # Distribution of picks by SKU class
        total_class_picks = sum(picks_by_class.values())
        if total_class_picks > 0:
            picks_by_class_pct = {k: (v / total_class_picks) * 100 for k, v in picks_by_class.items()}
        else:
            picks_by_class_pct = {k: 0.0 for k in picks_by_class}

        # Average distance to pick A-class SKUs (if available)
        # This requires knowing which picker_path points correspond to A-class picks; here we approximate
        avg_a_sku_pick_distance = None
        if a_skus:
            a_sku_distances = []
            for idx, batch_orders in enumerate(orders_by_picker):
                for order in batch_orders:
                    for line in order.lines:
                        if line.sku_id in a_skus:
                            # Approximate: use picker's total distance divided by lines picked
                            a_sku_distances.append(per_picker_distances[idx] / per_picker_kpis[idx]["Lines Picked"])
            avg_a_sku_pick_distance = sum(a_sku_distances) / len(a_sku_distances) if a_sku_distances else 0

        # Picker workload balance: variance in orders, lines, distance
        import statistics
        workload_balance = {
            "Orders Variance": statistics.variance(orders_per_picker) if len(orders_per_picker) > 1 else 0,
            "Lines Variance": statistics.variance([k["Lines Picked"] for k in per_picker_kpis]) if len(per_picker_kpis) > 1 else 0,
            "Distance Variance": statistics.variance(per_picker_distances) if len(per_picker_distances) > 1 else 0
        }

        # SKU-level frequency
        sku_pick_freq = sku_pick_counts

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
                "Average Batch Size": avg_batch_size,
                "SKU Overlap": sku_overlap,
                "Average Aisles Visited": avg_aisles_visited,
                "Average Penetration Depth": avg_penetration_depth,
                "Picks by Class (%)": picks_by_class_pct,
                "Average Distance to A-class SKU": avg_a_sku_pick_distance,
                "Picker Workload Balance": workload_balance,
                "SKU Pick Frequency": sku_pick_freq,
                "Pickers": len(picker_paths),
                "Total Orders": total_orders,
                "Total Lines Picked": total_lines
            }
        }
        return kpis