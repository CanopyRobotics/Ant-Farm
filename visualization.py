import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import itertools
import matplotlib.lines as mlines

def plot_warehouse_map(num_aisles, sections_per_side, wh, orders_by_picker, sku_to_location, picker_paths, kpis, a_skus, b_skus, c_skus):
    aisle_width = wh.aisle_width_m
    aisle_length = wh.aisle_length_m
    rack_depth = 0.5
    section_length = aisle_length / sections_per_side
    cross_aisle_height = 2.0

    x_positions = []
    for aisle in range(num_aisles):
        x_left_rack = aisle * (2 * rack_depth + aisle_width)
        x_aisle = x_left_rack + rack_depth
        x_right_rack = x_aisle + aisle_width
        x_positions.append((x_left_rack, x_aisle, x_right_rack))

    total_height = sections_per_side * section_length

    fig, ax = plt.subplots(figsize=(num_aisles * 2.5, sections_per_side * 1.2))
    plt.subplots_adjust(bottom=0.18)

    for aisle in range(num_aisles):
        x_left_rack, x_aisle, x_right_rack = x_positions[aisle]
        aisle_num = aisle + 1
        left_side_id = f"A{2*aisle_num-1:03d}"
        right_side_id = f"A{2*aisle_num:03d}"

        for side_id, x_rack, facecolor in [
            (left_side_id, x_left_rack, 'lightblue'),
            (right_side_id, x_right_rack, 'lightgreen')
        ]:
            for s_idx in range(sections_per_side):
                section_letters = chr(ord('A') + (s_idx // 26)) + chr(ord('A') + (s_idx % 26))
                y = s_idx * section_length
                rect = patches.Rectangle((x_rack, y), rack_depth, section_length, linewidth=1, edgecolor='black', facecolor=facecolor)
                ax.add_patch(rect)
                ax.text(x_rack + rack_depth / 2, y + section_length / 2, section_letters, ha='center', va='center', fontsize=7)

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
        x_positions[-1][2] + rack_depth + 2
    )
    ax.set_ylim(-cross_aisle_height - 0.5, total_height + cross_aisle_height + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.title("Warehouse Map (meters): Racks, Aisles, Sides, Picker Path", fontsize=14)

    # Picker paths and sliders
    colors = itertools.cycle(['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    path_lines = []
    path_dots = []
    sliders = []
    ax_slider_list = []

    for i, path in enumerate(picker_paths):
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        color = next(colors)
        line, = ax.plot(xs, ys, '-', color=color, linewidth=2, alpha=0.7, label=f'Picker {i+1}')
        dots, = ax.plot(xs, ys, 'o', color=color, markersize=5, alpha=0.7)
        path_lines.append(line)
        path_dots.append(dots)

        ax_slider = plt.axes([0.15, 0.01 + 0.05 * i, 0.7, 0.03])
        slider = Slider(ax_slider, f'Picker {i+1} Step', 1, len(path), valinit=len(path), valstep=1)
        sliders.append(slider)
        ax_slider_list.append(ax_slider)

        def make_update_func(line=line, dots=dots, path=path):
            def update(val):
                step = int(val)
                xs = [p[0] for p in path[:step]]
                ys = [p[1] for p in path[:step]]
                line.set_data(xs, ys)
                dots.set_data(xs, ys)
                plt.draw()
            return update

        slider.on_changed(make_update_func())

    # Legend handles (only up to min picker time KPIs)
    handles = []
    picker_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    def underline(text):
        return ''.join([c + '\u0332' for c in text])

    for i, picker_kpi in enumerate(kpis["Per Picker"]):
        picker_label = f'Picker {picker_kpi["Picker"]}'
        picker_label_fmt = f'\N{ZERO WIDTH SPACE}{underline(picker_label)}'
        line = mlines.Line2D([], [], color=picker_colors[i], label=picker_label_fmt, linewidth=2)
        kpi_label = (
            f'Orders Completed: {picker_kpi["Orders Completed"]}\n'
            f'Distance: {picker_kpi["Distance Walked (m)"]:.2f} m\n'
            f'Time: {picker_kpi["Time (s)"]:.2f} s ({picker_kpi["Time (min)"]:.2f} min)'
        )
        kpi_handle = mlines.Line2D([], [], color='none', label='   ' + kpi_label)
        handles.append(line)
        handles.append(kpi_handle)

    # Operation-wide KPIs (only up to and including Min Picker Time (s))
    op_kpi_title = f'\N{ZERO WIDTH SPACE}{underline("Operation-wide KPIs")}'
    op_kpi_text = op_kpi_title + "\n"
    op = kpis.get("Operation", {})
    ordered_keys = [
        "Total Distance Walked (m)",
        "Total Time (min)",
        "Average Picker Distance (m)",
        "Average Picker Time (min)",
        "Max Picker Distance (m)",
        "Min Picker Distance (m)",
        "Max Picker Time (s)",
        "Min Picker Time (s)",
    ]
    for k in ordered_keys:
        if k in op:
            v = op[k]
            if isinstance(v, (int, float)):
                op_kpi_text += f"{k}: {v:.2f}\n"
            else:
                op_kpi_text += f"{k}: {v}\n"
    handles.append(mlines.Line2D([], [], color='none', label=op_kpi_text))

    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(-0.4, 1), fontsize=9, frameon=True, title="Pickers & KPIs", handlelength=2)

    plt.show()