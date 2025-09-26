import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import kpis

def plot_warehouse_map(num_aisles, sections_per_side, wh, orders_by_picker, sku_to_location, picker_path, kpis):
    aisle_width = wh.aisle_width_m
    aisle_length = wh.aisle_length_m
    rack_depth = 0.5
    section_length = aisle_length / sections_per_side
    cross_aisle_height = 2.0

    # Compute X positions for each aisle and rack
    x_positions = []
    for aisle in range(num_aisles):
        x_left_rack = aisle * (2 * rack_depth + aisle_width)
        x_aisle = x_left_rack + rack_depth
        x_right_rack = x_aisle + aisle_width
        x_positions.append((x_left_rack, x_aisle, x_right_rack))

    total_height = sections_per_side * section_length

    # Draw warehouse
    fig, ax = plt.subplots(figsize=(num_aisles * 2.5, sections_per_side * 1.2))
    plt.subplots_adjust(bottom=0.18)

    # Draw racks and labels
    for aisle in range(num_aisles):
        x_left_rack, x_aisle, x_right_rack = x_positions[aisle]
        aisle_num = aisle + 1
        left_side_id = f"A{2*aisle_num-1:03d}"
        right_side_id = f"A{2*aisle_num:03d}"

        for s_idx in range(sections_per_side):
            section_letters = chr(ord('A') + (s_idx // 26)) + chr(ord('A') + (s_idx % 26))
            y = s_idx * section_length
            rect = patches.Rectangle((x_left_rack, y), rack_depth, section_length, linewidth=1, edgecolor='black', facecolor='lightblue')
            ax.add_patch(rect)
            ax.text(x_left_rack + rack_depth / 2, y + section_length / 2, section_letters, ha='center', va='center', fontsize=7)

        for s_idx in range(sections_per_side):
            section_letters = chr(ord('A') + (s_idx // 26)) + chr(ord('A') + (s_idx % 26))
            y = s_idx * section_length
            rect = patches.Rectangle((x_right_rack, y), rack_depth, section_length, linewidth=1, edgecolor='black', facecolor='lightgreen')
            ax.add_patch(rect)
            ax.text(x_right_rack + rack_depth / 2, y + section_length / 2, section_letters, ha='center', va='center', fontsize=7)

        # Place labels above the top cross-aisle
        label_y = total_height + cross_aisle_height + 0.5
        ax.text(
            x_aisle + aisle_width / 2,
            label_y,
            f"Aisle {aisle_num}",
            ha='center', va='bottom', fontsize=10, color='navy', rotation=90
        )
        ax.text(
            x_left_rack + rack_depth / 2,
            label_y + 0.5,
            left_side_id,
            ha='center', va='bottom', fontsize=9, color='blue', fontweight='bold', rotation=90
        )
        ax.text(
            x_right_rack + rack_depth / 2,
            label_y + 0.5,
            right_side_id,
            ha='center', va='bottom', fontsize=9, color='green', fontweight='bold', rotation=90
        )

    # Draw cross-aisle at bottom
    ax.add_patch(
        patches.Rectangle(
            (x_positions[0][0] - rack_depth, -cross_aisle_height),
            x_positions[-1][2] + rack_depth - (x_positions[0][0] - rack_depth),
            cross_aisle_height,
            linewidth=0,
            facecolor='#f0e68c',
            alpha=0.5,
            zorder=0
        )
    )
    ax.text(x_positions[0][0] + 1, -cross_aisle_height / 2, "Cross-aisle", va='center', ha='left', fontsize=10, color='brown')

    # Draw cross-aisle at top
    ax.add_patch(
        patches.Rectangle(
            (x_positions[0][0] - rack_depth, total_height),
            x_positions[-1][2] + rack_depth - (x_positions[0][0] - rack_depth),
            cross_aisle_height,
            linewidth=0,
            facecolor='#f0e68c',
            alpha=0.5,
            zorder=0
        )
    )
    ax.text(x_positions[0][0] + 1, total_height + cross_aisle_height / 2, "Cross-aisle", va='center', ha='left', fontsize=10, color='brown')

    ax.set_xlim(
        x_positions[0][0] - rack_depth - 0.5,
        x_positions[-1][2] + rack_depth + 0.5
    )
    ax.set_ylim(-cross_aisle_height - 0.5, total_height + cross_aisle_height + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("Warehouse Map (meters): Racks, Aisles, Sides, Picker Path", fontsize=14)

    # Add KPI metrics
    def format_kpi(k, v):
        if "Time" in k or "time" in k or "Walk" in k or "Pick" in k:
            return f"{k}: {v:.2f} s ({v/60:.2f} min)"
        elif isinstance(v, float):
            return f"{k}: {v:.2f}"
        else:
            return f"{k}: {v}"
    
    kpi_text = "\n".join(format_kpi(k, v) for k, v in kpis.items())
    ax.legend([kpi_text], loc='upper left', bbox_to_anchor=(-0.28 , 1), fontsize=10, frameon=True, title="KPIs")
    
    # Add slider for picker path
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.04])
    slider = Slider(ax_slider, 'Step', 1, max(1, len(picker_path)), valinit=1, valstep=1)

    path_line, = ax.plot([], [], '-', color='red', linewidth=2, alpha=0.7)
    path_dots, = ax.plot([], [], 'o', color='red', markersize=5, alpha=0.7)

    def update(val):
        step = int(slider.val)
        xs = [p[0] for p in picker_path[:step]]
        ys = [p[1] for p in picker_path[:step]]
        path_line.set_data(xs, ys)
        path_dots.set_data(xs, ys)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(1)

    plt.show()