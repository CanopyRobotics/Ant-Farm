import os
import dash
from dash import dcc, html
from dash.dependencies import Input
import pandas as pd
import plotly.express as px

# ---- ADD STYLE VARIABLES HERE ----
section_style = {
    "background": "#fafafa",
    "border": "1px solid #e0e0e0",
    "borderRadius": "8px",
    "padding": "18px 24px",
    "marginBottom": "24px",
    "boxShadow": "0 1px 2px rgba(0,0,0,0.06)"
}
table_style = {
    "marginBottom": "20px",
    "width": "100%",
    "borderCollapse": "collapse",
    "fontSize": "16px"
}
th_style = {"textAlign": "left", "padding": "8px", "fontWeight": "bold"}
td_style = {"padding": "8px"}
td_right = {"textAlign": "right", "padding": "8px"}
# ---- END STYLE VARIABLES ----

# Load latest CSV from runallcombos_results
RESULTS_DIR = "runallcombos_results"
csv_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")] if os.path.isdir(RESULTS_DIR) else []
csv_files.sort(reverse=True)
CSV_PATH = os.path.join(RESULTS_DIR, csv_files[0]) if csv_files else None

DF = pd.read_csv(CSV_PATH) if CSV_PATH else pd.DataFrame()

# Ensure numeric types for key metrics
for col in ["Total Distance Walked (m)", "Average Picker Time (min)"]:
    if col in DF.columns:
        DF[col] = pd.to_numeric(DF[col], errors="coerce")

app = dash.Dash(__name__)
app.css.append_css({
    "external_url": "https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css"
})

app.layout = html.Div([
    html.H1("Warehouse Policy Experiment Dashboard"),
    html.Div(id="summary-section"),
    html.Div(id="best-policies-section"),
    html.Div(id="top-combos-section"),
    html.Hr(),
    html.H2("What-if: Gains vs Standard Baseline"),
    html.Div(id="what-if-cards", style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
    html.Div(id="comparative-table", style={"marginBottom": "18px"}),
    # --- SLOTING PLOTS ---
    dcc.Graph(id="slotting-barplot"),
    dcc.Graph(id="slotting-time-boxplot"),
    dcc.Graph(id="slotting-distance-barplot"),
    dcc.Graph(id="slotting-distance-boxplot"),
    dcc.Graph(id="slotting-scatterplot"),
    # --- BATCHING PLOTS ---
    dcc.Graph(id="picker-time-barplot"),
    # --- ROUTING PLOTS ---
    dcc.Graph(id="distance-boxplot"),
    dcc.Graph(id="distance-vs-time-scatter"),
], style={"width": "100%", "display": "inline-block", "paddingLeft": "2%"}),


@app.callback(
    [
        dash.Output("summary-section", "children"),
        dash.Output("best-policies-section", "children"),
        dash.Output("top-combos-section", "children"),
        dash.Output("what-if-cards", "children"),
        dash.Output("comparative-table", "children"),
        dash.Output("slotting-barplot", "figure"),
        dash.Output("slotting-time-boxplot", "figure"),
        dash.Output("slotting-distance-barplot", "figure"),
        dash.Output("slotting-distance-boxplot", "figure"),
        dash.Output("slotting-scatterplot", "figure"),
        dash.Output("picker-time-barplot", "figure"),
        dash.Output("distance-boxplot", "figure"),
        dash.Output("distance-vs-time-scatter", "figure"),
    ],
    [Input("summary-section", "id")]
)
def update_graphs(_):
    if DF.empty:
        empty_msg = html.Div("No data found. Run experiments to generate a CSV in runallcombos_results.")
        fig_empty = px.scatter(pd.DataFrame(columns=["x", "y"]), x="x", y="y", title="No data")
        return (
            empty_msg,         # summary-section
            html.Div(),        # best-policies-section
            html.Div(),        # top-combos-section
            [],                # what-if-cards
            html.Div(),        # comparative-table
            fig_empty,         # slotting-barplot
            fig_empty,         # slotting-time-boxplot
            fig_empty,         # slotting-distance-barplot
            fig_empty,         # slotting-distance-boxplot
            fig_empty,         # slotting-scatterplot
            fig_empty,         # picker-time-barplot
            fig_empty,         # distance-boxplot
            fig_empty          # distance-vs-time-scatter
        )

    # Best policies by type (using lowest avg picker time)
    slot_group = DF.groupby("Slotting")["Average Picker Time (min)"].mean()
    batch_group = DF.groupby("Batching")["Average Picker Time (min)"].mean()
    route_group = DF.groupby("Routing")["Average Picker Time (min)"].mean()
    best_slotting = slot_group.idxmin(); best_slotting_val = float(slot_group.min())
    best_batching = batch_group.idxmin(); best_batching_val = float(batch_group.min())
    best_routing = route_group.idxmin(); best_routing_val = float(route_group.min())

    # Top 3 overall combinations
    combo_cols = ["Slotting", "Batching", "Routing"]
    combo_group = DF.groupby(combo_cols)["Average Picker Time (min)"].mean().reset_index()
    top_combos = combo_group.nsmallest(3, "Average Picker Time (min)")

    # Summary cards section
    summary_section = html.Div([
        html.H3("Best Policies (by Average Picker Time [min])"),
        html.Div([
            html.Div([
                html.H4("Best Slotting"),
                html.P(f"{best_slotting} ({best_slotting_val:.2f} min)")
            ], style={"textAlign": "center"}),
            html.Div([
                html.H4("Best Batching"),
                html.P(f"{best_batching} ({best_batching_val:.2f} min)")
            ], style={"textAlign": "center"}),
            html.Div([
                html.H4("Best Routing"),
                html.P(f"{best_routing} ({best_routing_val:.2f} min)")
            ], style={"textAlign": "center"}),
        ], style={"display": "flex", "gap": "24px", "justifyContent": "space-between"})
    ], style=section_style)

    # Best policies by type (using lowest avg picker time)
    slot_group = DF.groupby("Slotting")["Average Picker Time (min)"].mean()
    batch_group = DF.groupby("Batching")["Average Picker Time (min)"].mean()
    route_group = DF.groupby("Routing")["Average Picker Time (min)"].mean()
    best_slotting = slot_group.idxmin(); best_slotting_val = float(slot_group.min())
    best_batching = batch_group.idxmin(); best_batching_val = float(batch_group.min())
    best_routing = route_group.idxmin(); best_routing_val = float(route_group.min())

    # Top 3 overall combinations
    combo_cols = ["Slotting", "Batching", "Routing"]
    combo_group = DF.groupby(combo_cols)["Average Picker Time (min)"].mean().reset_index()
    top_combos = combo_group.nsmallest(3, "Average Picker Time (min)")

    # Summary cards section
    summary_section = html.Div([
        html.H3("Best Policies (by Average Picker Time [min])"),
        html.Div([
            html.Div([
                html.H4("Best Slotting"),
                html.P(f"{best_slotting} ({best_slotting_val:.2f} min)")
            ], style={"textAlign": "center"}),
            html.Div([
                html.H4("Best Batching"),
                html.P(f"{best_batching} ({best_batching_val:.2f} min)")
            ], style={"textAlign": "center"}),
            html.Div([
                html.H4("Best Routing"),
                html.P(f"{best_routing} ({best_routing_val:.2f} min)")
            ], style={"textAlign": "center"}),
        ], style={"display": "flex", "gap": "24px", "justifyContent": "space-between"})
    ], style=section_style)

    # Best policies table section
    best_policies_section = html.Div([
        html.H4("Best Policy Table"),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Policy Type", style=th_style),
                html.Th("Policy Name", style=th_style),
                html.Th("Avg Picker Time (min)", style={**th_style, **td_right})
            ])),
            html.Tbody([
                html.Tr([
                    html.Td("Slotting", style=td_style),
                    html.Td(best_slotting, style=td_style),
                    html.Td(f"{best_slotting_val:.2f}", style=td_right)
                ]),
                html.Tr([
                    html.Td("Batching", style=td_style),
                    html.Td(best_batching, style=td_style),
                    html.Td(f"{best_batching_val:.2f}", style=td_right)
                ]),
                html.Tr([
                    html.Td("Routing", style=td_style),
                    html.Td(best_routing, style=td_style),
                    html.Td(f"{best_routing_val:.2f}", style=td_right)
                ]),
            ])
        ], style=table_style)
    ], style=section_style)

    # Top combos table section
    top_combos_section = html.Div([
        html.H4("Top 3 Combinations (by Lowest Avg Picker Time)"),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Slotting", style=th_style),
                html.Th("Batching", style=th_style),
                html.Th("Routing", style=th_style),
                html.Th("Avg Picker Time (min)", style={**th_style, **td_right})
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(row["Slotting"], style=td_style),
                    html.Td(row["Batching"], style=td_style),
                    html.Td(row["Routing"], style=td_style),
                    html.Td(f"{row['Average Picker Time (min)']:.2f}", style=td_right)
                ]) for _, row in top_combos.iterrows()
            ])
        ], style=table_style)
    ], style=section_style)

    # ===== What-if vs Standard (RandomSlotting + RandomBatching + SShapeRouting) =====
    def mean_time(df_sub: pd.DataFrame | None):
        if df_sub is None or df_sub.empty:
            return None
        s = pd.to_numeric(df_sub["Average Picker Time (min)"], errors="coerce").dropna()
        return float(s.mean()) if len(s) else None

    standard_combo = {"Slotting": "RandomSlotting", "Batching": "RandomBatching", "Routing": "SShapeRouting"}
    base_df = DF[(DF["Slotting"] == standard_combo["Slotting"]) &
                 (DF["Batching"] == standard_combo["Batching"]) &
                 (DF["Routing"] == standard_combo["Routing"])]
    base_time = mean_time(base_df)

    # Best Slotting-only (keep batching & routing at standard)
    slot_df = DF[(DF["Batching"] == standard_combo["Batching"]) & (DF["Routing"] == standard_combo["Routing"])].copy()
    slot_rank = slot_df.groupby("Slotting")["Average Picker Time (min)"].mean().sort_values() if not slot_df.empty else pd.Series(dtype=float)
    best_slot_name = slot_rank.index[0] if len(slot_rank) else None
    best_slot_time = float(slot_rank.iloc[0]) if len(slot_rank) else None

    # Best Batching-only (keep slotting & routing at standard)
    batch_df = DF[(DF["Slotting"] == standard_combo["Slotting"]) & (DF["Routing"] == standard_combo["Routing"])]
    batch_rank = batch_df.groupby("Batching")["Average Picker Time (min)"].mean().sort_values() if not batch_df.empty else pd.Series(dtype=float)
    best_batch_name = batch_rank.index[0] if len(batch_rank) else None
    best_batch_time = float(batch_rank.iloc[0]) if len(batch_rank) else None

    # Best Routing-only (keep slotting & batching at standard)
    route_df = DF[(DF["Slotting"] == standard_combo["Slotting"]) & (DF["Batching"] == standard_combo["Batching"])]
    route_rank = route_df.groupby("Routing")["Average Picker Time (min)"].mean().sort_values() if not route_df.empty else pd.Series(dtype=float)
    best_route_name = route_rank.index[0] if len(route_rank) else None
    best_route_time = float(route_rank.iloc[0]) if len(route_rank) else None

    # Best overall combination
    combo_rank = combo_group.sort_values("Average Picker Time (min)") if not combo_group.empty else pd.DataFrame()
    best_combo_row = combo_rank.iloc[0] if not combo_rank.empty else None
    best_combo_time = float(best_combo_row["Average Picker Time (min)"]) if best_combo_row is not None else None
    best_combo_names = (
        (best_combo_row["Slotting"], best_combo_row["Batching"], best_combo_row["Routing"]) if best_combo_row is not None else (None, None, None)
    )

    def pct_gain(base, new):
        if base is None or new is None or base == 0:
            return None
        return (base - new) / base * 100.0

    gain_slot = pct_gain(base_time, best_slot_time)
    gain_batch = pct_gain(base_time, best_batch_time)
    gain_route = pct_gain(base_time, best_route_time)
    gain_combo = pct_gain(base_time, best_combo_time)

    def card(title, value_pct, subtitle):
        color = ("#2e7d32" if (value_pct is not None and value_pct >= 0) else "#c62828")
        val_str = (f"{value_pct:.1f}%" if value_pct is not None else "N/A")
        return html.Div([
            html.Div(title, style={"fontWeight": "600", "fontSize": "14px", "marginBottom": "6px"}),
            html.Div(val_str, style={"fontSize": "24px", "fontWeight": "700", "color": color}),
            html.Div(subtitle, style={"fontSize": "12px", "opacity": 0.8})
        ], style={
            "flex": "1 1 220px", "background": "#fafafa", "border": "1px solid #e0e0e0",
            "borderRadius": "8px", "padding": "12px 16px", "boxShadow": "0 1px 2px rgba(0,0,0,0.06)"
        })

    what_if_cards = [
        card(
            "Slotting-only gain",
            gain_slot,
            f"Best: {best_slot_name or 'N/A'} vs Standard ({(base_time or 0):.2f} min)" if base_time is not None else "No baseline"
        ),
        card(
            "Batching-only gain",
            gain_batch,
            f"Best: {best_batch_name or 'N/A'} vs Standard ({(base_time or 0):.2f} min)" if base_time is not None else "No baseline"
        ),
        card(
            "Routing-only gain",
            gain_route,
            f"Best: {best_route_name or 'N/A'} vs Standard ({(base_time or 0):.2f} min)" if base_time is not None else "No baseline"
        ),
        card(
            "All policies gain",
            gain_combo,
            (f"Best Combo: {best_combo_names[0]} | {best_combo_names[1]} | {best_combo_names[2]}\n"
             f"{(best_combo_time or 0):.2f} min vs Standard {(base_time or 0):.2f} min")
            if base_time is not None and best_combo_time is not None else "No baseline or combo"
        )
    ]

    # Comparative table
    comp_rows: list[list[str]] = []
    if base_time is not None:
        comp_rows.append(["Baseline (RandomSlotting, RandomBatching, SShapeRouting)", f"{base_time:.2f}", "—"])
    if best_slot_name and best_slot_time is not None:
        comp_rows.append([f"Slotting: {best_slot_name} (others standard)", f"{best_slot_time:.2f}", f"{(gain_slot or 0):.1f}%"])
    if best_batch_name and best_batch_time is not None:
        comp_rows.append([f"Batching: {best_batch_name} (others standard)", f"{best_batch_time:.2f}", f"{(gain_batch or 0):.1f}%"])
    if best_route_name and best_route_time is not None:
        comp_rows.append([f"Routing: {best_route_name} (others standard)", f"{best_route_time:.2f}", f"{(gain_route or 0):.1f}%"])
    if best_combo_row is not None and best_combo_time is not None:
        comp_rows.append([f"Best Combo: {best_combo_names[0]} | {best_combo_names[1]} | {best_combo_names[2]}", f"{best_combo_time:.2f}", f"{(gain_combo or 0):.1f}%"])

    comparative_table = html.Table([
        html.Thead(html.Tr([html.Th("Scenario"), html.Th("Avg Picker Time (min)"), html.Th("% Gain vs Baseline")])),
        html.Tbody([
            html.Tr([html.Td(r[0]), html.Td(r[1]), html.Td(r[2])]) for r in comp_rows
        ])
    ], style={"width": "100%", "borderCollapse": "collapse"})

    # --- Plots ---
    # Barplot: Average Picker Time by Slotting Policy
    fig_slotting_bar = px.bar(
        DF.groupby("Slotting")["Average Picker Time (min)"].mean().reset_index(),
        x="Slotting", y="Average Picker Time (min)",
        title="Average Picker Time by Slotting Policy"
    )
    fig_slotting_bar.update_layout(template="plotly_white", height=360)

    # Boxplot: Picker Time by Slotting Policy
    fig_slotting_time_box = px.box(
        DF, x="Slotting", y="Average Picker Time (min)",
        title="Distribution of Picker Time by Slotting Policy"
    )
    fig_slotting_time_box.update_layout(template="plotly_white", height=360)

    # Barplot: Average Total Distance Walked by Slotting Policy
    fig_slotting_dist_bar = px.bar(
        DF.groupby("Slotting")["Total Distance Walked (m)"].mean().reset_index(),
        x="Slotting", y="Total Distance Walked (m)",
        title="Average Total Distance Walked by Slotting Policy"
    )
    fig_slotting_dist_bar.update_layout(template="plotly_white", height=360)

    # Boxplot: Total Distance Walked by Slotting Policy
    fig_slotting_dist_box = px.box(
        DF, x="Slotting", y="Total Distance Walked (m)",
        title="Distribution of Distance Walked by Slotting Policy"
    )
    fig_slotting_dist_box.update_layout(template="plotly_white", height=360)

    # Scatterplot: Distance vs Picker Time by Slotting Policy
    fig_slotting_scatter = px.scatter(
        DF, x="Total Distance Walked (m)", y="Average Picker Time (min)",
        color="Slotting", symbol="Slotting",
        title="Distance vs Picker Time by Slotting Policy",
        hover_data=["Batching", "Routing", "Iteration"] if "Iteration" in DF.columns else ["Batching", "Routing"]
    )
    fig_slotting_scatter.update_layout(template="plotly_white", legend_title_text="Slotting Policy", height=420)
    fig_slotting_scatter.update_traces(marker=dict(size=10), selector=dict(mode="markers"))

    # Barplot: Average Picker Time by Batching
    avg_times = DF.groupby("Batching")["Average Picker Time (min)"].mean().reset_index()
    fig2 = px.bar(avg_times, x="Batching", y="Average Picker Time (min)", title="Avg Picker Time by Batching Policy")
    fig2.update_layout(template="plotly_white", height=360)

    # Boxplot: Show all Routing policies
    fig1 = px.box(DF, x="Routing", y="Total Distance Walked (m)", title="Total Distance by Routing Policy")
    fig1.update_layout(template="plotly_white", height=360)

    # Scatterplot: All policies, colored by Routing
    fig3 = px.scatter(
        DF,
        x="Total Distance Walked (m)",
        y="Average Picker Time (min)",
        color="Routing", symbol="Routing",
        title="Distance vs Picker Time by Routing",
        hover_data=["Slotting", "Batching", "Iteration"] if "Iteration" in DF.columns else ["Slotting", "Batching"]
    )
    fig3.update_layout(template="plotly_white", legend_title_text="Routing Policy", height=420)
    fig3.update_traces(marker=dict(size=10), selector=dict(mode="markers"))

    return (
        summary_section,
        best_policies_section,
        top_combos_section,
        what_if_cards,
        comparative_table,
        # --- SLOTING PLOTS ---
        fig_slotting_bar, fig_slotting_time_box, fig_slotting_dist_bar, fig_slotting_dist_box, fig_slotting_scatter,
        # --- BATCHING PLOTS ---
        fig2,
        # --- ROUTING PLOTS ---
        fig1, fig3
    )
    gain_slot = pct_gain(base_time, best_slot_time)
    gain_batch = pct_gain(base_time, best_batch_time)
    gain_route = pct_gain(base_time, best_route_time)
    gain_combo = pct_gain(base_time, best_combo_time)

    def card(title, value_pct, subtitle):
        color = ("#2e7d32" if (value_pct is not None and value_pct >= 0) else "#c62828")
        val_str = (f"{value_pct:.1f}%" if value_pct is not None else "N/A")
        return html.Div([
            html.Div(title, style={"fontWeight": "600", "fontSize": "14px", "marginBottom": "6px"}),
            html.Div(val_str, style={"fontSize": "24px", "fontWeight": "700", "color": color}),
            html.Div(subtitle, style={"fontSize": "12px", "opacity": 0.8})
        ], style={
            "flex": "1 1 220px", "background": "#fafafa", "border": "1px solid #e0e0e0",
            "borderRadius": "8px", "padding": "12px 16px", "boxShadow": "0 1px 2px rgba(0,0,0,0.06)"
        })

    what_if_cards = [
        card(
            "Slotting-only gain",
            gain_slot,
            f"Best: {best_slot_name or 'N/A'} vs Standard ({(base_time or 0):.2f} min)" if base_time is not None else "No baseline"
        ),
        card(
            "Batching-only gain",
            gain_batch,
            f"Best: {best_batch_name or 'N/A'} vs Standard ({(base_time or 0):.2f} min)" if base_time is not None else "No baseline"
        ),
        card(
            "Routing-only gain",
            gain_route,
            f"Best: {best_route_name or 'N/A'} vs Standard ({(base_time or 0):.2f} min)" if base_time is not None else "No baseline"
        ),
        card(
            "All policies gain",
            gain_combo,
            (f"Best Combo: {best_combo_names[0]} | {best_combo_names[1]} | {best_combo_names[2]}\n"
             f"{(best_combo_time or 0):.2f} min vs Standard {(base_time or 0):.2f} min")
            if base_time is not None and best_combo_time is not None else "No baseline or combo"
        )
    ]

    # Comparative table
    comp_rows: list[list[str]] = []
    if base_time is not None:
        comp_rows.append(["Baseline (RandomSlotting, RandomBatching, SShapeRouting)", f"{base_time:.2f}", "—"])
    if best_slot_name and best_slot_time is not None:
        comp_rows.append([f"Slotting: {best_slot_name} (others standard)", f"{best_slot_time:.2f}", f"{(gain_slot or 0):.1f}%"])
    if best_batch_name and best_batch_time is not None:
        comp_rows.append([f"Batching: {best_batch_name} (others standard)", f"{best_batch_time:.2f}", f"{(gain_batch or 0):.1f}%"])
    if best_route_name and best_route_time is not None:
        comp_rows.append([f"Routing: {best_route_name} (others standard)", f"{best_route_time:.2f}", f"{(gain_route or 0):.1f}%"])
    if best_combo_row is not None and best_combo_time is not None:
        comp_rows.append([f"Best Combo: {best_combo_names[0]} | {best_combo_names[1]} | {best_combo_names[2]}", f"{best_combo_time:.2f}", f"{(gain_combo or 0):.1f}%"])

    comparative_table = html.Table([
        html.Thead(html.Tr([html.Th("Scenario"), html.Th("Avg Picker Time (min)"), html.Th("% Gain vs Baseline")])),
        html.Tbody([
            html.Tr([html.Td(r[0]), html.Td(r[1]), html.Td(r[2])]) for r in comp_rows
        ])
    ], style={"width": "100%", "borderCollapse": "collapse"})

    # --- Plots ---
    # Boxplot: Show all Routing policies
    fig1 = px.box(DF, x="Routing", y="Total Distance Walked (m)", title="Total Distance by Routing Policy")
    fig1.update_layout(template="plotly_white", height=360)
    if DF.empty:
        fig1.add_annotation(text="No data for this filter", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16))

    # Barplot: Average Picker Time by Batching
    avg_times = DF.groupby("Batching")["Average Picker Time (min)"].mean().reset_index()
    fig2 = px.bar(avg_times, x="Batching", y="Average Picker Time (min)", title="Avg Picker Time by Batching Policy")
    fig2.update_layout(template="plotly_white", height=360)
    if avg_times.empty:
        fig2.add_annotation(text="No data for this filter", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16))

    # Scatterplot: All policies, colored by Routing
    fig3 = px.scatter(
        DF,
        x="Total Distance Walked (m)",
        y="Average Picker Time (min)",
        color="Routing", symbol="Routing",
        title="Distance vs Picker Time by Routing",
        hover_data=["Slotting", "Batching", "Iteration"] if "Iteration" in DF.columns else ["Slotting", "Batching"]
    )
    fig3.update_layout(template="plotly_white", legend_title_text="Routing Policy", height=420)
    fig3.update_traces(marker=dict(size=10), selector=dict(mode="markers"))

    # Barplot: Average Picker Time by Slotting Policy
    fig_slotting_bar = px.bar(
        DF.groupby("Slotting")["Average Picker Time (min)"].mean().reset_index(),
        x="Slotting", y="Average Picker Time (min)",
        title="Average Picker Time by Slotting Policy"
    )
    fig_slotting_bar.update_layout(template="plotly_white", height=360)

    # Boxplot: Picker Time by Slotting Policy
    fig_slotting_time_box = px.box(
        DF, x="Slotting", y="Average Picker Time (min)",
        title="Distribution of Picker Time by Slotting Policy"
    )
    fig_slotting_time_box.update_layout(template="plotly_white", height=360)

    # Barplot: Average Total Distance Walked by Slotting Policy
    fig_slotting_dist_bar = px.bar(
        DF.groupby("Slotting")["Total Distance Walked (m)"].mean().reset_index(),
        x="Slotting", y="Total Distance Walked (m)",
        title="Average Total Distance Walked by Slotting Policy"
    )
    fig_slotting_dist_bar.update_layout(template="plotly_white", height=360)

    # Boxplot: Total Distance Walked by Slotting Policy
    fig_slotting_dist_box = px.box(
        DF, x="Slotting", y="Total Distance Walked (m)",
        title="Distribution of Distance Walked by Slotting Policy"
    )
    fig_slotting_dist_box.update_layout(template="plotly_white", height=360)

    # Scatterplot: Distance vs Picker Time by Slotting Policy
    fig_slotting_scatter = px.scatter(
        DF, x="Total Distance Walked (m)", y="Average Picker Time (min)",
        color="Slotting", symbol="Slotting",
        title="Distance vs Picker Time by Slotting Policy",
        hover_data=["Batching", "Routing", "Iteration"] if "Iteration" in DF.columns else ["Batching", "Routing"]
    )
    fig_slotting_scatter.update_layout(template="plotly_white", legend_title_text="Slotting Policy", height=420)
    fig_slotting_scatter.update_traces(marker=dict(size=10), selector=dict(mode="markers"))

    return (
        summary_section,
        best_policies_section,
        top_combos_section,
        # --- SLOTING PLOTS ---
        fig_slotting_bar, fig_slotting_time_box, fig_slotting_dist_bar, fig_slotting_dist_box, fig_slotting_scatter,
        # --- BATCHING PLOTS ---
        fig2,
        # --- ROUTING PLOTS ---
        fig1, fig3
    )


if __name__ == "__main__":
    app.run(debug=True)