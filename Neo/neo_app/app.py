import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import io, base64
from dataclasses import make_dataclass

# Try package-relative imports; fall back to local when run as script
try:
    from .data_io import read_layout, read_sku_locations, read_sales
    from .policies import AffinitySlotting, PopularityABCSlotting, RandomSlotting, RoundRobinSlotting
    from affinity import compute_affinity_matrix as af_compute_affinity_matrix, group_skus_by_affinity as af_group_skus_by_affinity
    from models import Order
    from .visuals import plot_floorplan_with_heat, compute_vmin_vmax
except ImportError:
    from data_io import read_layout, read_sku_locations, read_sales
    from policies import AffinitySlotting, PopularityABCSlotting, RandomSlotting, RoundRobinSlotting
    from affinity import compute_affinity_matrix as af_compute_affinity_matrix, group_skus_by_affinity as af_group_skus_by_affinity
    from models import Order
    from visuals import plot_floorplan_with_heat, compute_vmin_vmax

# Simple styles
CARD = {"border": "1px solid #e0e0e0", "borderRadius": "8px", "padding": "12px", "marginBottom": "12px", "background": "#fff"}

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Neo – Warehouse Slotting Optimizer"),
    html.Div([
        html.Div([
            html.H3("Upload Files"),
            dcc.Upload(id='upload-layout', children=html.Div(['Drag & Drop layout.csv or ', html.A('Select File')]), multiple=False, style=CARD),
            html.Div(id="upload-layout-status", style={"marginBottom": "8px", "fontSize": "12px", "color": "#333"}),
            dcc.Upload(id='upload-sku', children=html.Div(['Drag & Drop sku_locations.csv or ', html.A('Select File')]), multiple=False, style=CARD),
            html.Div(id="upload-sku-status", style={"marginBottom": "8px", "fontSize": "12px", "color": "#333"}),
            dcc.Upload(id='upload-sales', children=html.Div(['Drag & Drop sales.csv or ', html.A('Select File')]), multiple=False, style=CARD),
            html.Div(id="upload-sales-status", style={"marginBottom": "8px", "fontSize": "12px", "color": "#333"}),
            html.Label("Policy"),
            dcc.Dropdown(id='policy', options=[
                {"label": "Affinity Slotting", "value": "affinity"},
                {"label": "ABC Slotting", "value": "abc"},
                {"label": "Round Robin", "value": "rr"},
                {"label": "Random", "value": "rand"},
            ], value="affinity", style={"marginBottom": "12px"}),
            html.Button("Compute Proposal", id="run", n_clicks=0)
        ], style={"flex": 1}),
        html.Div([
            html.H3("Heatmaps — Current vs Proposed"),
            html.Div([
                dcc.Loading(dcc.Graph(id="heatmap-current"), type="dot", style={"width": "48%", "display": "inline-block"}),
                dcc.Loading(dcc.Graph(id="heatmap-proposed"), type="dot", style={"width": "48%", "display": "inline-block", "float": "right"}),
            ])
        ], style={"flex": 2})
    ], style={"display": "flex", "gap": "16px"})
,
    # fewer moving parts, no hidden stores
])


def parse_upload(contents) -> pd.DataFrame:
    if contents is None:
        return None
    content_type, content_string = contents.split(',')
    decoded = io.BytesIO(base64.b64decode(content_string))
    return pd.read_csv(decoded)


def _as_loc_dict(loc_obj):
    # supports dataclass or dict
    if hasattr(loc_obj, "__dict__"):
        d = loc_obj.__dict__
        return {"side_id": d["side_id"], "section": d["section"], "tote_id": d["tote_id"]}
    return {"side_id": loc_obj["side_id"], "section": loc_obj["section"], "tote_id": loc_obj["tote_id"]}

def _enforce_one_sku_per_tote(assign_map, layout_df, sku_ids):
    """
    Ensure a 1:1 mapping (one SKU per tote). If collisions occur, move overflow to the next free tote
    in layout reading order. Assumes len(sku_ids) == number of totes (100% occupancy).
    """
    loc_cols = ["side_id", "section", "tote_id"]
    # reading order = layout order
    free = list(layout_df["tote_id"])
    taken = set()
    resolved = {}

    def take_tote(tid):
        if tid in taken:
            return None
        taken.add(tid)
        free.remove(tid)
        return tid

    # First pass: try to keep assigned tote if unique
    for sku in sorted(sku_ids):
        loc_obj = assign_map.get(sku)
        if loc_obj is None:
            # will assign in second pass
            resolved[sku] = None
            continue
        loc = _as_loc_dict(loc_obj)
        tid = loc["tote_id"]
        if tid in free:
            take_tote(tid)
            resolved[sku] = loc
        else:
            resolved[sku] = None  # will be placed in second pass

    # Second pass: place any unresolved SKUs into remaining free totes
    # Use layout_df to pull the full location tuple for the tote
    idx_by_tote = layout_df.set_index("tote_id")[["side_id", "section"]].to_dict("index")
    for sku in sorted(sku_ids):
        if resolved[sku] is None:
            if not free:
                raise RuntimeError("Not enough totes to enforce 1 SKU per tote.")
            tid = free[0]
            take_tote(tid)
            s = idx_by_tote[tid]["side_id"]
            sec = idx_by_tote[tid]["section"]
            resolved[sku] = {"side_id": s, "section": sec, "tote_id": tid}

    return resolved


@app.callback(
    Output("upload-layout-status", "children"),
    Output("upload-sku-status", "children"),
    Output("upload-sales-status", "children"),
    Input("upload-layout", "contents"),
    Input("upload-sku", "contents"),
    Input("upload-sales", "contents"),
)
def update_upload_status(layout_contents, sku_contents, sales_contents):
    """Return small summaries/previews for each uploaded CSV to give visual confirmation."""
    def preview(contents):
        if contents is None:
            return "No file uploaded."
        try:
            df = parse_upload(contents)
            if df is None:
                return "No file uploaded."
            rows, cols = df.shape
            # show up to 5 rows as a small CSV text
            preview_csv = df.head(5).to_csv(index=False)
            # keep it short
            preview_lines = preview_csv.splitlines()
            preview_text = "\n".join(preview_lines[:10])
            return html.Div([
                html.Div(f"Uploaded — {rows} rows x {cols} cols", style={"fontWeight": "600"}),
                html.Pre(preview_text, style={"whiteSpace": "pre-wrap", "maxHeight": "160px", "overflow": "auto", "fontSize": "11px"})
            ])
        except Exception as e:
            return html.Div([html.Div("Failed to parse file", style={"color": "red"}), html.Div(str(e), style={"fontSize": "11px", "color": "#555"})])

    return preview(layout_contents), preview(sku_contents), preview(sales_contents)


@app.callback(
    Output("heatmap-current", "figure"),
    Output("heatmap-proposed", "figure"),
    Input("run", "n_clicks"),
    State("upload-layout", "contents"),
    State("upload-sku", "contents"),
    State("upload-sales", "contents"),
    State("policy", "value"),
    prevent_initial_call=True
)
def run_policy(n, layout_contents, sku_contents, sales_contents, policy):
    layout_df = parse_upload(layout_contents)
    sku_df = parse_upload(sku_contents)
    sales_df = parse_upload(sales_contents)

    if layout_df is None or sku_df is None:
        return {}, {}

    # Ensure expected columns
    for col in ["tote_id", "side_id", "section"]:
        if col not in layout_df.columns:
            return {}, {}
    if "sku_id" not in sku_df.columns or "tote_id" not in sku_df.columns:
        return {}, {}

    # Build current map (1 SKU per tote assumed in input)
    current_map = sku_df.merge(layout_df[["tote_id", "side_id", "section"]], on="tote_id", how="left")

    # Prepare locations in layout reading order
    loc_cols = ["side_id", "section", "tote_id"]
    StorageLocation = make_dataclass("StorageLocation", [(c, str) for c in loc_cols])
    locations_objs = [StorageLocation(**row) for row in layout_df[loc_cols].to_dict("records")]
    # Build unique, sorted list of SKUs to assign
    sku_ids = pd.Series(current_map["sku_id"]).dropna().astype(int).unique().tolist()
    sku_ids.sort()
    # If more SKUs than totes, trim to available totes; prefer highest sales if sales_df present
    if len(sku_ids) > len(locations_objs):
        if sales_df is not None and "sku_id" in sales_df.columns and "quantity" in sales_df.columns:
            sales_rank = sales_df.groupby("sku_id")["quantity"].sum().sort_values(ascending=False)
            top_skus = [int(s) for s in sales_rank.index.tolist() if int(s) in set(sku_ids)][:len(locations_objs)]
            if len(top_skus) < len(locations_objs):
                # fill remaining with leftover SKUs
                remaining = [s for s in sku_ids if s not in set(top_skus)]
                top_skus.extend(remaining[:len(locations_objs)-len(top_skus)])
            sku_ids = top_skus
        else:
            sku_ids = sku_ids[:len(locations_objs)]

    # Choose policy (default to affinity)
    try:
        from .policies import AffinitySlotting, PopularityABCSlotting, RandomSlotting, RoundRobinSlotting
    except ImportError:
        from policies import AffinitySlotting, PopularityABCSlotting, RandomSlotting, RoundRobinSlotting

    if policy == "affinity":
        if sales_df is None or "order_id" not in sales_df.columns:
            # Fallback: treat as ABC if no sales/orders
            pop = getattr(sales_df.groupby("sku_id")["quantity"], "sum", lambda: pd.Series(dtype=int))().to_dict() if sales_df is not None else {}
            policy_obj = PopularityABCSlotting(pop)
            assign = policy_obj.assign(sku_ids, locations_objs)
        else:
            # Build Ant-Farm Order objects from sales
            order_groups = sales_df.groupby("order_id")["sku_id"].apply(list)
            orders = [Order(id=int(oid), lines=[type("OL", (), {"sku_id": int(s)})() for s in skus]) for oid, skus in order_groups.items()]
            affinity = af_compute_affinity_matrix(orders)
            groups = af_group_skus_by_affinity(affinity, sku_ids, group_size=18)
            policy_obj = AffinitySlotting(groups)
            assign = policy_obj.assign(sku_ids, locations_objs)
    elif policy == "abc":
        pop = sales_df.groupby("sku_id")["quantity"].sum().to_dict() if sales_df is not None else {}
        policy_obj = PopularityABCSlotting(pop)
        assign = policy_obj.assign(sku_ids, locations_objs)
    elif policy == "rr":
        assign = RoundRobinSlotting().assign(sku_ids, locations_objs)
    else:
        assign = RandomSlotting().assign(sku_ids, locations_objs)

    # Enforce 1 SKU per tote deterministically (no vacancies, full 1:1)
    assign_1to1 = _enforce_one_sku_per_tote(assign, layout_df, sku_ids)

    proposed_map = pd.DataFrame([{"sku_id": sku, **assign_1to1[sku]} for sku in sku_ids])

    # Render thermal floorplan overlays (red=hot, blue=cold)
    # Compute shared vmin/vmax so colors are comparable across current & proposed
    vmin, vmax = compute_vmin_vmax(layout_df, [current_map, proposed_map], sales_df)
    heat_current = plot_floorplan_with_heat(
        layout_df, current_map, sales_df, title="Current: Warehouse Heat Overlay", colorscale="RdYlBu_r", vmin=vmin, vmax=vmax
    )
    heat_proposed = plot_floorplan_with_heat(
        layout_df, proposed_map, sales_df, title="Proposed: Warehouse Heat Overlay", colorscale="RdYlBu_r", vmin=vmin, vmax=vmax
    )
    return heat_current, heat_proposed


# Removed front-view and selectors per request to simplify


if __name__ == "__main__":
    # use the new Dash API
    app.run(debug=True)
