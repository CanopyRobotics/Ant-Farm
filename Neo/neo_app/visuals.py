import plotly.express as px
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from typing import Dict

def plot_before_after(current_map: pd.DataFrame, proposed_map: pd.DataFrame):
    # current_map/proposed_map: columns [sku_id, side_id, section, tote_id]
    current_map = current_map.assign(state="Current")
    proposed_map = proposed_map.assign(state="Proposed")
    df = pd.concat([current_map, proposed_map], ignore_index=True)
    fig = px.treemap(
        df,
        path=["state", "side_id", "section", "tote_id"],
        values=None,
        color="state",
        hover_data=["sku_id"],
        color_discrete_map={"Current": "#95a5a6", "Proposed": "#27ae60"}
    )
    fig.update_layout(margin=dict(t=30, l=10, r=10, b=10))
    return fig


def plot_heatmap(layout_df: pd.DataFrame, sku_locations_df: pd.DataFrame, sales_df: pd.DataFrame):
    """Build a warehouse heatmap of sales quantity per (aisle, side, section).
    layout_df: must contain columns [side_id, section, tote_id]
    The heatmap uses the full layout to fix aisles/sections and shows empty totes as zeros.
    """
    # Sum sales quantity per sku (if provided)
    sku_qty = pd.DataFrame(columns=["sku_id", "qty"]) if sales_df is None else sales_df.groupby("sku_id").quantity.sum().reset_index().rename(columns={"quantity": "qty"})

    # Build a full mapping of totes from layout and attach sku/tote and qty where present
    merged_all = layout_df.merge(sku_locations_df, on="tote_id", how="left")
    merged_all = merged_all.merge(sku_qty, on="sku_id", how="left")
    merged_all["qty"] = merged_all["qty"].fillna(0)

    # Parse side_id -> aisle and side index based on layout_df (ensures consistent grid)
    def parse_side(side_id: str):
        try:
            n = int(side_id[1:])
            aisle = (n + 1) // 2
            side = 'L' if (n % 2) == 1 else 'R'
            return aisle, side
        except Exception:
            return None, None

    # Map section string to short code (AA -> 'AA') and index
    def short_section(s):
        if isinstance(s, str) and '-' in s:
            return s.split('-')[-1]
        return s

    def section_index(s: str):
        if not isinstance(s, str) or len(s) == 0:
            return 0
        s = s.upper()
        letters = ''.join([c for c in s if c.isalpha()])[:2]
        if len(letters) == 1:
            return ord(letters[0]) - ord('A')
        return (ord(letters[0]) - ord('A')) * 26 + (ord(letters[1]) - ord('A'))

    # derive max_aisle and max_sidx from layout_df
    layout_sections = layout_df.copy()
    layout_sections["_section"] = layout_sections["section"].apply(short_section)
    layout_sections["_aisle"] = layout_sections["side_id"].apply(lambda s: parse_side(str(s))[0])
    layout_sections["_sidx"] = layout_sections["_section"].apply(section_index)

    max_aisle = int(layout_sections["_aisle"].max()) if layout_sections["_aisle"].notna().any() else 0
    max_sidx = int(layout_sections["_sidx"].max()) if layout_sections["_sidx"].notna().any() else 0

    if max_aisle == 0 or max_sidx == 0:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(text="Insufficient layout data to build heatmap", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white")
        return fig

    # include a center walking space column between left and right sides for each aisle
    cols = max_aisle * 3  # left / walking / right per aisle
    rows = max_sidx + 1

    import numpy as np
    grid = np.zeros((rows, cols))

    def col_index(aisle, side):
        if aisle is None or side is None:
            return None
        base = (int(aisle) - 1) * 3
        return base + (0 if side == 'L' else 2)

    # populate grid using merged_all which contains every tote in layout
    for _, r in merged_all.iterrows():
        aisle_side = parse_side(str(r.get("side_id", "")))
        aisle = aisle_side[0]
        side = aisle_side[1]
        s = section_index(short_section(r.get("section", "")))
        c = col_index(aisle, side)
        if c is None or s is None:
            continue
        if 0 <= s < rows and 0 <= c < cols:
            grid[s, c] += float(r.get("qty", 0))

    # make walking gap columns visually empty (NaN) so they appear as aisle space
    for aisle in range(1, max_aisle + 1):
        gap_idx = (aisle - 1) * 3 + 1
        if 0 <= gap_idx < cols:
            grid[:, gap_idx] = np.nan

    # Create axis labels with walking gaps
    x_labels = []
    for aisle in range(1, max_aisle + 1):
        x_labels.append(f"A{aisle:03d}-L")
        x_labels.append("")
        x_labels.append(f"A{aisle:03d}-R")

    y_labels = []
    for s in range(rows):
        if s < 26:
            y_labels.append(chr(ord('A') + s))
        else:
            first = s // 26
            second = s % 26
            y_labels.append(chr(ord('A') + first) + chr(ord('A') + second))

    # Use log scale for color to make hotspots visible
    display = grid.copy()
    with np.errstate(divide='ignore'):
        display = np.log1p(display)

    fig = px.imshow(display[::-1, :],  # flip rows so A at bottom
                    x=x_labels,
                    y=y_labels[::-1],
                    color_continuous_scale='YlOrRd',
                    labels={'x': 'Aisle/Side', 'y': 'Section (letters)', 'color': 'log(quantity+1)'} )
    fig.update_layout(title="Sales Heatmap by Aisle/Side/Section", margin=dict(t=40, l=60))
    return fig


def plot_birdseye_overlay(layout_df: pd.DataFrame, map_df: pd.DataFrame, sales_df: pd.DataFrame,
                          aisle_width_m: float = 2.0, aisle_length_m: float = 20.0, rack_depth: float = 0.5):
    """Draw a clear bird's-eye floorplan (aisles and rack faces) and overlay a heat color per section.
    - Aggregates values per section (sum of SKU quantities from sales_df when provided, otherwise counts SKUs)
    - Draws left/right rack rectangles per aisle and colors them according to the aggregated metric.
    Returns a Plotly figure with explicit shapes so walking aisles are obvious.
    """
    import plotly.graph_objects as go
    import numpy as np

    # coerce inputs to DataFrame if they're records from dcc.Store
    if not isinstance(layout_df, pd.DataFrame):
        try:
            layout_df = pd.DataFrame(layout_df)
        except Exception:
            layout_df = None
    if not isinstance(map_df, pd.DataFrame):
        try:
            map_df = pd.DataFrame(map_df)
        except Exception:
            map_df = None

    if layout_df is None or map_df is None:
        fig = go.Figure(); fig.add_annotation(text="Upload layout and sku map to see bird's-eye map", xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)
        return fig

    # Prepare mapping tote -> section grouping
    # Section id is the portion of 'section' (e.g. A003-AB -> AB)
    def section_short(s):
        if not isinstance(s, str):
            return s
        return s.split('-')[-1]

    layout = layout_df.copy()
    layout['_section_short'] = layout['section'].apply(section_short)

    # estimate sections_per_side
    sections_per_side = int(layout.groupby('side_id')['_section_short'].nunique().median()) if not layout.empty else 10
    section_length = aisle_length_m / max(1, sections_per_side)

    # compute quantity per section by summing sales per SKU then mapping to the section via map_df
    if sales_df is None:
        # count SKUs per section
        merged = map_df.merge(layout[['tote_id', 'section']], on='tote_id', how='left')
        # ensure 'section' column exists after merge
        if 'section' not in merged.columns:
            merged = merged.merge(layout[['tote_id', 'section']], on='tote_id', how='left')
        merged['_section_short'] = merged['section'].apply(lambda x: section_short(x) if pd.notna(x) else x)
        sec_qty = merged.groupby(['section', '_section_short'])['sku_id'].nunique().reset_index().rename(columns={'sku_id': 'qty'})
    else:
        sku_qty = sales_df.groupby('sku_id')['quantity'].sum().reset_index().rename(columns={'quantity': 'qty'})
        merged = map_df.merge(sku_qty, on='sku_id', how='left')
        # ensure 'section' exists, merge from layout if needed
        if 'section' not in merged.columns:
            merged = merged.merge(layout[['tote_id', 'section']], on='tote_id', how='left')
        merged['qty'] = merged['qty'].fillna(0)
        merged['_section_short'] = merged['section'].apply(lambda x: section_short(x) if pd.notna(x) else x)
        sec_qty = merged.groupby(['section', '_section_short'])['qty'].sum().reset_index()

    # Build bird's-eye shapes per aisle and per side
    # parse side_id like A003 where numeric digits indicate side index
    def parse_side(side_id):
        try:
            n = int(side_id[1:])
            aisle = (n + 1) // 2
            side = 'L' if (n % 2) == 1 else 'R'
            return aisle, side
        except Exception:
            return None, None

    # get unique aisles sorted
    aisles = sorted({parse_side(sid)[0] for sid in layout['side_id'] if parse_side(sid)[0] is not None})
    max_aisle_idx = max(aisles) if aisles else 1

    # compute x positions for left/right racks using same scheme as location_to_xy
    # aisle indices: 0-based internal calculation
    coords = []
    for _, r in layout.iterrows():
        side_id = r['side_id']
        try:
            n = int(side_id[1:])
        except Exception:
            continue
        aisle_idx = n // 2
        is_left = (n % 2) == 1
        x_left_rack = aisle_idx * (2 * rack_depth + aisle_width_m)
        x_aisle = x_left_rack + rack_depth
        x_right_rack = x_aisle + aisle_width_m
        x_center = x_left_rack + rack_depth / 2 if is_left else x_right_rack + rack_depth / 2
        s_short = section_short(r['section'])
        s_idx = 0
        # compute section index from letters
        if isinstance(s_short, str) and len(s_short) > 0:
            letters = ''.join([c for c in s_short if c.isalpha()])[:2]
            if len(letters) == 1:
                s_idx = ord(letters[0]) - ord('A')
            elif len(letters) == 2:
                s_idx = (ord(letters[0]) - ord('A')) * 26 + (ord(letters[1]) - ord('A'))
        y_center = s_idx * section_length + section_length / 2
        coords.append({'tote_id': r['tote_id'], 'section': r['section'], 'section_short': s_short, 'aisle_idx': aisle_idx, 'is_left': is_left, 'x': x_center, 'y': y_center})

    cdf = pd.DataFrame(coords)

    # Map section-level qty
    sec_qty_map = {row['section']: row['qty'] for _, row in sec_qty.iterrows()}
    max_qty = max([v for v in sec_qty_map.values()]) if sec_qty_map else 0

    shapes = []
    annotations = []

    # For each unique section and aisle side, draw rectangles
    # We'll iterate over aisles and sections indices
    x_positions = {}
    # create mapping from (aisle_idx, 'L'/'R') -> x_start
    for _, r in cdf.iterrows():
        key = (r['aisle_idx'], 'L' if r['is_left'] else 'R')
        x_positions[key] = r['x']

    # create sorted y centers
    y_centers = sorted(cdf['y'].unique())

    # build color mapping (simple linear to rgba)
    def qty_to_color(q):
        if max_qty <= 0:
            return 'rgba(200,200,200,0.3)'
        frac = float(q) / float(max_qty)
        # color ramp from light yellow to red
        r = int(255 * frac)
        g = int(200 * (1 - frac))
        b = 50
        a = 0.85 if frac > 0 else 0.05
        return f'rgba({r},{g},{b},{a})'

    # for each unique section (section id), compute its y range and draw left/right if exist
    sections = sorted(cdf['section'].unique())
    for sec in sections:
        rows = cdf[cdf['section'] == sec]
        if rows.empty:
            continue
        # pick a representative y
        y = rows.iloc[0]['y']
        top = y - section_length / 2
        bottom = y + section_length / 2
        qty = sec_qty_map.get(sec, 0)
        color = qty_to_color(qty)
        # draw for each aisle-side present
        for _, r in rows.iterrows():
            x_center = r['x']
            x0 = x_center - rack_depth / 2
            x1 = x_center + rack_depth / 2
            shapes.append(dict(type='rect', x0=x0, x1=x1, y0=top, y1=bottom, fillcolor=color, line=dict(width=0)))
            annotations.append(dict(x=x_center, y=y, text=f"{sec}\n{int(qty)}", showarrow=False, font=dict(size=9), xanchor='center', yanchor='middle'))

    fig = go.Figure()
    fig.update_layout(shapes=shapes)
    # add annotations
    for ann in annotations:
        fig.add_annotation(ann)

    fig.update_xaxes(title_text='X (m)')
    fig.update_yaxes(title_text='Y (m)')
    fig.update_layout(title='Birds-eye Floorplan with Section Heat Overlay', margin=dict(t=40), height=600)
    return fig


def _aisle_num(side_id: str) -> int:
    """Return 1-based aisle number from a side_id like 'A001' or 'A002'.
    A001/A002 -> aisle 1, A003/A004 -> aisle 2, etc."""
    try:
        n = int(str(side_id)[1:])
        # A001/A002 -> aisle 1, A003/A004 -> aisle 2
        return (n + 1) // 2
    except Exception:
        return 1


def _section_indexer(sections: pd.Series) -> tuple[dict, list]:
    """Build a mapping from section letter suffix (e.g., 'AB') to a 0-based index, sorted naturally.
    Returns (index_map, ordered_list)."""
    def suffix(s):
        s = str(s)
        return s.split('-')[-1] if '-' in s else s
    def key(letters: str):
        letters = str(letters)
        if len(letters) == 0:
            return (9999, 9999)
        # support 1-2 letters
        if len(letters) == 1:
            return (0, ord(letters[0].upper()) - ord('A'))
        return (ord(letters[0].upper()) - ord('A') + 1, ord(letters[1].upper()) - ord('A'))
    uniq_letters = sorted({suffix(s) for s in sections.dropna().unique().tolist()}, key=key)
    return {l: i for i, l in enumerate(uniq_letters)}, uniq_letters


def plot_floorplan_with_heat(layout_df: pd.DataFrame,
                             map_df: pd.DataFrame,
                             sales_df: pd.DataFrame | None,
                             title: str = "",
                             colorscale: str = "RdYlBu_r",
                             rack_depth: float = 1.0,
                             aisle_width: float = 2.0,
                             section_length: float = 1.0,
                             vmin: float | None = None,
                             vmax: float | None = None,
                             tote_heat_override: Dict[str, float] | None = None) -> go.Figure:
    """
    Draw Ant-Farm-like floorplan and color each rack-section rectangle by traffic.
    - Traffic is sum(quantity) per SKU from sales_df, aggregated to the section where the SKU is stored (via map_df).
    - The floorplan is fixed by layout_df; only colors change between current vs proposed.
    """
    layout_df = pd.DataFrame(layout_df).copy()
    map_df = pd.DataFrame(map_df).copy()

    # Ensure map_df has side_id/section via tote_id
    if "section" not in map_df.columns or "side_id" not in map_df.columns:
        map_df = map_df.merge(layout_df[["tote_id", "side_id", "section"]], on="tote_id", how="left")

    # Compute per-section heat. If tote_heat_override is provided, aggregate per-tote picks to section-level.
    if tote_heat_override is not None:
        # Build mapping tote -> section letters
        tm = layout_df[["tote_id", "side_id", "section"]].copy()
        tm["sec_letters"] = tm["section"].astype(str).apply(lambda s: s.split('-')[-1] if '-' in s else s)
        tm["qty"] = tm["tote_id"].map(lambda t: float(tote_heat_override.get(t, 0.0)))
        sec_qty = tm.groupby(["side_id", "sec_letters"], dropna=False)["qty"].sum().reset_index()
    else:
        if sales_df is not None and "sku_id" in sales_df.columns and "quantity" in sales_df.columns:
            sku_qty = sales_df.groupby("sku_id")["quantity"].sum()
        else:
            sku_qty = pd.Series(1, index=pd.Index(map_df["sku_id"].unique(), name="sku_id"))
        merged = map_df[["side_id", "section", "sku_id"]].copy()
        merged["qty"] = merged["sku_id"].map(sku_qty).fillna(0)
        merged["sec_letters"] = merged["section"].astype(str).apply(lambda s: s.split('-')[-1] if '-' in s else s)
        sec_qty = merged.groupby(["side_id", "sec_letters"], dropna=False)["qty"].sum().reset_index()

    # Geometry: derive aisles and sections from layout
    # A001 is left, A002 is right of aisle 1; include walking space between them.
    sides = layout_df[["side_id"]].drop_duplicates().sort_values("side_id")
    aisles = sorted({_aisle_num(sid) for sid in sides["side_id"]})
    sec_index, sec_order = _section_indexer(layout_df["section"])  # index by letter suffix

    fig = go.Figure()
    # Use log scaling and percentile clipping for better contrast; respect provided vmin/vmax
    z_vals = sec_qty["qty"].to_numpy() if len(sec_qty) else np.array([0.0])
    z_log = np.log1p(z_vals)
    if vmin is None or vmax is None:
        nz = z_log[z_vals > 0]
        if nz.size > 0:
            vmin_calc = float(np.percentile(nz, 5))
            vmax_calc = float(np.percentile(nz, 95))
            if vmax_calc <= vmin_calc:
                vmax_calc = vmin_calc + 1.0
        else:
            vmin_calc, vmax_calc = 0.0, 1.0
        vmin = vmin if vmin is not None else vmin_calc
        vmax = vmax if vmax is not None else vmax_calc

    def color_for(val: float) -> str:
        t = (np.log1p(val) - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, float(t)))
        return sample_colorscale(colorscale, t)[0]

    # Draw per-section rectangles using aggregated section heat
    shapes = []
    # quick lookup for section quantities per side
    by_side = {}
    for sid, grp in sec_qty.groupby("side_id"):
        by_side[sid] = grp.set_index("sec_letters")["qty"].to_dict()
    for aisle in aisles:
        x_left_rack = (aisle - 1) * (2 * rack_depth + aisle_width)
        x_right_rack = x_left_rack + rack_depth + aisle_width
        for side, x_rack in (("L", x_left_rack), ("R", x_right_rack)):
            sid = f"A{(2*aisle-1 if side=='L' else 2*aisle):03d}"
            side_sec = by_side.get(sid, {})
            for sec_letters, idx in sec_index.items():
                y0 = idx * section_length
                y1 = y0 + section_length
                qty = float(side_sec.get(sec_letters, 0.0))
                shapes.append(dict(
                    type="rect",
                    x0=x_rack, x1=x_rack + rack_depth,
                    y0=y0, y1=y1,
                    line=dict(width=0.5, color="rgba(50,50,50,0.25)"),
                    fillcolor=color_for(qty)
                ))
        # walking lane separators
        shapes.append(dict(
            type="rect",
            x0=x_left_rack + rack_depth, x1=x_right_rack,
            y0=0, y1=len(sec_order) * section_length,
            line=dict(width=0), fillcolor="rgba(255,255,255,0.95)"
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        shapes=shapes,
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    return fig


def compute_vmin_vmax(layout_df: pd.DataFrame,
                      map_dfs: list[pd.DataFrame],
                      sales_df: pd.DataFrame | None,
                      lower_pct: float = 5.0,
                      upper_pct: float = 95.0) -> tuple[float, float]:
    """Compute shared log-space vmin/vmax across one or more mapping DataFrames.
    This ensures current and proposed heatmaps share the same color normalization.
    Returns (vmin, vmax) in log1p space.
    """
    all_logs = []
    layout_df = pd.DataFrame(layout_df)
    for map_df in map_dfs:
        m = pd.DataFrame(map_df).copy()
        if "section" not in m.columns or "side_id" not in m.columns:
            m = m.merge(layout_df[["tote_id", "side_id", "section"]], on="tote_id", how="left")
        # per-SKU qty
        if sales_df is not None and "sku_id" in sales_df.columns and "quantity" in sales_df.columns:
            sku_qty = sales_df.groupby("sku_id")["quantity"].sum()
        else:
            sku_qty = pd.Series(1, index=pd.Index(m["sku_id"].unique(), name="sku_id"))
        m["qty"] = m["sku_id"].map(sku_qty).fillna(0)
        merged = m[["side_id", "section", "qty"]].copy()
        merged["sec_letters"] = merged["section"].astype(str).apply(lambda s: s.split('-')[-1] if '-' in s else s)
        sec_qty = merged.groupby(["side_id", "sec_letters"], dropna=False)["qty"].sum().reset_index()
        z = sec_qty["qty"].to_numpy() if len(sec_qty) else np.array([], dtype=float)
        if z.size:
            all_logs.append(np.log1p(z[z > 0]))
    if not all_logs:
        return 0.0, 1.0
    stacked = np.concatenate(all_logs) if len(all_logs) > 1 else all_logs[0]
    vmin = float(np.percentile(stacked, lower_pct))
    vmax = float(np.percentile(stacked, upper_pct))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def plot_heatmap_for_assignment(layout_df: pd.DataFrame, map_df: pd.DataFrame, sales_df: pd.DataFrame):
    """Build the same heatmap but where sku->location mapping is taken from map_df (sku_id -> tote_id/side/section).
    map_df must contain columns [sku_id, tote_id, side_id, section]."""
    import plotly.graph_objects as go
    if layout_df is None or map_df is None or sales_df is None:
        fig = go.Figure()
        fig.add_annotation(text="Upload layout, sku locations and sales to see heatmap", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white")
        return fig

    sku_qty = sales_df.groupby("sku_id").quantity.sum().reset_index().rename(columns={"quantity": "qty"})
    merged = map_df.merge(sku_qty, on="sku_id", how="left")
    merged["qty"] = merged["qty"].fillna(0)

    # For sections we only need side/section parsing same as plot_heatmap
    def parse_side(side_id: str):
        try:
            n = int(side_id[1:])
            aisle = (n + 1) // 2
            side = 'L' if (n % 2) == 1 else 'R'
            return aisle, side
        except Exception:
            return None, None

    aisles = []
    sides = []
    sections = []
    for _, row in merged.iterrows():
        aisle, side = parse_side(str(row.get("side_id", "")))
        aisles.append(aisle)
        sides.append(side)
        sec = row.get("section", "")
        if isinstance(sec, str) and '-' in sec:
            sec = sec.split('-')[-1]
        sections.append(sec)

    merged = merged.assign(_aisle=aisles, _side=sides, _section=sections)

    def section_index(s: str):
        if not isinstance(s, str) or len(s) == 0:
            return 0
        s = s.upper()
        letters = ''.join([c for c in s if c.isalpha()])[:2]
        if len(letters) == 1:
            return ord(letters[0]) - ord('A')
        return (ord(letters[0]) - ord('A')) * 26 + (ord(letters[1]) - ord('A'))

    merged["_sidx"] = merged["_section"].apply(section_index)

    max_aisle = int(merged["_aisle"].max()) if merged["_aisle"].notna().any() else 0
    max_sidx = int(merged["_sidx"].max()) if merged["_sidx"].notna().any() else 0

    if max_aisle == 0 or max_sidx == 0:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient layout data to build heatmap", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white")
        return fig

    # include walking space between sides
    cols = max_aisle * 3
    rows = max_sidx + 1

    import numpy as np
    grid = np.zeros((rows, cols))

    def col_index(aisle, side):
        if aisle is None or side is None:
            return None
        base = (int(aisle) - 1) * 3
        return base + (0 if side == 'L' else 2)

    for _, r in merged.iterrows():
        c = col_index(r.get("_aisle"), r.get("_side"))
        s = int(r.get("_sidx", 0))
        if c is None or s is None:
            continue
        if 0 <= s < rows and 0 <= c < cols:
            grid[s, c] += float(r.get("qty", 0))

    x_labels = []
    for aisle in range(1, max_aisle + 1):
        x_labels.append(f"A{aisle:03d}-L")
        x_labels.append("")
        x_labels.append(f"A{aisle:03d}-R")

    y_labels = []
    for s in range(rows):
        if s < 26:
            y_labels.append(chr(ord('A') + s))
        else:
            first = s // 26
            second = s % 26
            y_labels.append(chr(ord('A') + first) + chr(ord('A') + second))

    display = grid.copy()
    with np.errstate(divide='ignore'):
        display = np.log1p(display)

    import plotly.express as px
    fig = px.imshow(display[::-1, :], x=x_labels, y=y_labels[::-1], color_continuous_scale='YlOrRd', labels={'x': 'Aisle/Side', 'y': 'Section (letters)', 'color': 'log(quantity+1)'} )
    fig.update_layout(title="Sales Heatmap (assigned mapping)", margin=dict(t=40, l=60))
    return fig


def plot_section_front_view(layout_df: pd.DataFrame, map_df: pd.DataFrame, sales_df: pd.DataFrame, section_id: str, highlight_tote: str = None):
    """Front view of a section (totes arranged by level and position). Returns a Plotly figure.
    section_id should be the full section like 'A003-AB'."""
    import plotly.graph_objects as go
    import numpy as np

    if layout_df is None or map_df is None:
        fig = go.Figure()
        fig.add_annotation(text="Upload data to preview section", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white")
        return fig

    # Use layout_df to list all totes for this section so empty positions are present
    section_layout = layout_df[layout_df["section"] == section_id]
    if section_layout.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No section {section_id} in layout", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white")
        return fig

    # Merge sku mapping (map_df) to layout to include empty totes
    section_map = section_layout.merge(map_df, on="tote_id", how="left")
    if section_map.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No totes found for {section_id}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white")
        return fig

    # Aggregate sales qty per sku -> per tote
    if sales_df is None:
        sku_qty = pd.DataFrame(columns=["sku_id", "qty"])
    else:
        sku_qty = sales_df.groupby("sku_id").quantity.sum().reset_index().rename(columns={"quantity": "qty"})

    merged = section_map.merge(sku_qty, on="sku_id", how="left")
    merged["qty"] = merged["qty"].fillna(0)

    # Determine level and position from tote_id last numeric part: tote_num = int(last)
    # Ant-Farm uses 3 positions per level; we will enforce 3 columns per level (pos 0..2)
    def tote_pos(tote_id: str):
        try:
            last = tote_id.split("-")[-1]
            n = int(last)
            level = (n - 1) // 10 + 1
            pos = (n - 1) % 10  # original numbering; we'll mod to map into columns
            # Map positions into 3 columns: use pos % 3
            col = pos % 3
            return level, col
        except Exception:
            return None, None

    levels = []
    poss = []
    totes = []
    for _, r in merged.iterrows():
        l, p = tote_pos(r["tote_id"])
        levels.append(l if l is not None else 0)
        poss.append(p if p is not None else 0)
        totes.append(r["tote_id"])

    merged = merged.assign(_level=levels, _pos=poss)

    max_level = int(merged["_level"].max()) if merged["_level"].notna().any() else 0
    # enforce 3 positions per level (cols = 3)
    rows = max_level
    cols = 3
    grid = np.zeros((rows, cols))
    labels = [['' for _ in range(cols)] for __ in range(rows)]

    # Sum qty per tote
    tote_qty = merged.groupby("tote_id").qty.sum().to_dict()

    for tote in merged["tote_id"].unique():
        part = merged[merged["tote_id"] == tote].iloc[0]
        l = int(part["_level"]) - 1
        p = int(part["_pos"])
        if p is None:
            continue
        if 0 <= l < rows and 0 <= p < cols:
            grid[rows - 1 - l, p] = tote_qty.get(tote, 0)
            labels[rows - 1 - l][p] = f"{tote}\n{int(tote_qty.get(tote,0))}"
        # if highlight requested, add a border annotation
        if highlight_tote is not None and tote == highlight_tote:
            # annotate by adding a thick-text marker
            labels[rows - 1 - l][p] = f"*{tote}\n{int(tote_qty.get(tote,0))}*"

    if rows == 0 or cols == 0:
        fig = go.Figure()
        fig.add_annotation(text=f"No level/pos info for {section_id}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_white")
        return fig

    with np.errstate(divide='ignore'):
        display = np.log1p(grid)

    fig = go.Figure(data=go.Heatmap(z=display, x=[f"P{c+1}" for c in range(cols)], y=[f"L{r+1}" for r in range(rows)], colorscale='YlGnBu', hoverinfo='z'))
    # annotate tote ids and qtys
    for r in range(rows):
        for c in range(cols):
            text = labels[r][c]
            if text:
                fig.add_annotation(x=c, y=r, text=text, showarrow=False, font=dict(size=10), xref='x', yref='y')

    fig.update_layout(title=f"Front view: {section_id}", xaxis_title='Position', yaxis_title='Level (top=1)', yaxis_autorange='reversed')
    return fig


def plot_heatmap_xy(layout_df: pd.DataFrame, map_df: pd.DataFrame, sales_df: pd.DataFrame,
                    aisle_width_m: float = 2.0, aisle_length_m: float = 20.0, rack_depth: float = 0.5):
    """Build a bird's-eye heatmap using explicit x/y coordinates derived from layout_df.
    layout_df must contain columns [side_id, section, tote_id]. map_df should include sku->tote mapping (sku_id,tote_id,...)
    sales_df is optional; if provided, heat shows aggregated qty per tote, otherwise counts of SKUs per tote.
    The function returns a Plotly figure where x positions reflect real distances, so walking aisles appear correctly.
    """
    import numpy as np
    import plotly.graph_objects as go

    if layout_df is None or map_df is None:
        fig = go.Figure(); fig.add_annotation(text="Upload layout and sku map to see heatmap", xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)
        return fig

    # derive sections_per_side from layout: count unique section suffixes per side and take median
    def short_section(s):
        return s.split('-')[-1] if isinstance(s, str) and '-' in s else s

    layout = layout_df.copy()
    layout['_section_short'] = layout['section'].apply(short_section)
    # estimate sections_per_side as unique section codes per side_id
    sections_per_side = int(layout.groupby('side_id')['_section_short'].nunique().median()) if not layout.empty else 10

    # compute section index (0-based)
    def section_index(s: str):
        if not isinstance(s, str) or len(s) == 0:
            return 0
        s = s.upper()
        letters = ''.join([c for c in s if c.isalpha()])[:2]
        if len(letters) == 1:
            return ord(letters[0]) - ord('A')
        return (ord(letters[0]) - ord('A')) * 26 + (ord(letters[1]) - ord('A'))

    # compute per-tote coordinates using Ant-Farm scheme
    def parse_side(side_id: str):
        try:
            n = int(side_id[1:])
            aisle = (n + 1) // 2
            side = 'L' if (n % 2) == 1 else 'R'
            return aisle, side
        except Exception:
            return None, None

    section_length = aisle_length_m / max(1, sections_per_side)

    # aggregate qty per sku then per tote
    if sales_df is None:
        # count SKUs per tote from map_df
        tote_qty = map_df.groupby('tote_id')['sku_id'].nunique().to_dict()
    else:
        sku_qty = sales_df.groupby('sku_id')['quantity'].sum().reset_index().rename(columns={'quantity': 'qty'})
        merged = map_df.merge(sku_qty, on='sku_id', how='left')
        merged['qty'] = merged['qty'].fillna(0)
        tote_qty = merged.groupby('tote_id')['qty'].sum().to_dict()

    # prepare full layout with coordinates
    coords = []
    for _, r in layout.iterrows():
        side_id = r['side_id']
        aisle, side = parse_side(str(side_id))
        s_short = short_section(r['section'])
        s_idx = section_index(s_short)

        # x coordinates as in location_to_xy
        aisle_idx = int(side_id[1:]) // 2  # 0-based
        is_left = int(side_id[1:]) % 2 == 1
        x_left_rack = aisle_idx * (2 * rack_depth + aisle_width_m)
        x_aisle = x_left_rack + rack_depth
        x_right_rack = x_aisle + aisle_width_m
        x = x_left_rack + rack_depth / 2 if is_left else x_right_rack + rack_depth / 2

        y = s_idx * section_length + section_length / 2

        qty = float(tote_qty.get(r['tote_id'], 0))
        coords.append({'tote_id': r['tote_id'], 'x': x, 'y': y, 'qty': qty, 'aisle': aisle, 'side': side, 'section_short': s_short})

    dfc = pd.DataFrame(coords)

    if dfc.empty:
        fig = go.Figure(); fig.add_annotation(text="No layout/totes found", xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False)
        return fig

    # Build grid bins: x centers are unique sorted x values; but we want spacing to reflect actual distances
    x_centers = sorted(dfc['x'].unique())
    y_centers = sorted(dfc['y'].unique())

    # create 2D grid matching centers
    grid = np.zeros((len(y_centers), len(x_centers)))
    for _, r in dfc.iterrows():
        ix = x_centers.index(r['x'])
        iy = y_centers.index(r['y'])
        grid[iy, ix] += r['qty']

    # display as heatmap with real coordinate axes
    # compute x/y extents for edges
    def centers_to_edges(centers):
        # symmetric edges around centers
        edges = []
        for i, c in enumerate(centers):
            if i == 0:
                prev = c - (centers[i+1] - c)/2 if len(centers) > 1 else c-0.5
            else:
                prev = (centers[i-1] + c)/2
            edges.append(prev)
        # add last edge
        last = centers[-1] + (centers[-1] - edges[-1])
        edges.append(last)
        return edges

    x_edges = centers_to_edges(x_centers)
    y_edges = centers_to_edges(y_centers)

    # Use log scale for color
    with np.errstate(divide='ignore'):
        display = np.log1p(grid)

    fig = go.Figure(data=go.Heatmap(z=display[::-1, :], x=x_centers, y=y_centers[::-1], colorscale='YlOrRd', colorbar=dict(title='log(qty+1)')))
    fig.update_layout(title='Birds-eye Sales Heatmap (physical coords)', xaxis_title='X (m)', yaxis_title='Y (m)', margin=dict(t=40))
    return fig
