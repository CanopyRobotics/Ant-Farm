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
    # Empty-state guard with 13 outputs
    if DF.empty:
        empty_msg = html.Div("No data found. Run experiments to generate a CSV in runallcombos_results.")
        fig_empty = px.scatter(pd.DataFrame(columns=["x", "y"]))
        return (
            empty_msg, html.Div(), html.Div(), [], html.Div(),
            fig_empty, fig_empty, fig_empty, fig_empty, fig_empty,
            fig_empty, fig_empty, fig_empty
        )

    df = DF.copy()
    # Ensure numeric types
    for col in [
        "Average Picker Time (min)", "Average Picker Distance (m)",
        "Total Distance Walked (m)", "Total Time (min)",
        "Average Orders per Picker", "Average Lines per Order",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Best policies
    slot_group = df.groupby("Slotting")["Average Picker Time (min)"].mean()
    batch_group = df.groupby("Batching")["Average Picker Time (min)"].mean()
    route_group = df.groupby("Routing")["Average Picker Time (min)"].mean()
    best_slotting = slot_group.idxmin(); best_slotting_val = float(slot_group.min())
    best_batching = batch_group.idxmin(); best_batching_val = float(batch_group.min())
    best_routing = route_group.idxmin(); best_routing_val = float(route_group.min())

    # Top combos
    combo_cols = ["Slotting", "Batching", "Routing"]
    combo_group = df.groupby(combo_cols)["Average Picker Time (min)"].mean().reset_index()
    top_combos = combo_group.nsmallest(3, "Average Picker Time (min)")

    # Summary section (professional cards)
    summary_kpis = [
        ("Total Distance Walked", f"{df['Total Distance Walked (m)'].mean():,.0f} m" if 'Total Distance Walked (m)' in df else "-"),
        ("Total Time", f"{df['Total Time (min)'].mean():.1f} min" if 'Total Time (min)' in df else "-"),
        ("Avg Picker Time", f"{df['Average Picker Time (min)'].mean():.2f} min"),
        ("Avg Picker Distance", f"{df['Average Picker Distance (m)'].mean():.0f} m" if 'Average Picker Distance (m)' in df else "-"),
        ("Avg Orders/Picker", f"{df['Average Orders per Picker'].mean():.1f}" if 'Average Orders per Picker' in df else "-"),
        ("Avg Lines/Order", f"{df['Average Lines per Order'].mean():.1f}" if 'Average Lines per Order' in df else "-"),
    ]
    summary_section = html.Div([
        html.H2("Experiment Summary", style={"marginBottom": "12px", "color": "#2c3e50"}),
        html.Div([
            html.Div([
                html.H4(title, style={"marginBottom": "4px", "color": "#34495e"}),
                html.P(val, style={"fontSize": "20px", "fontWeight": "bold", "color": "#2980b9", "margin": "0"})
            ], style={
                "background": "#f4f6f8", "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(44,62,80,0.07)",
                "padding": "16px 18px", "minWidth": "160px", "textAlign": "center",
                "marginRight": "18px", "marginBottom": "12px"
            }) for title, val in summary_kpis
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "0"})
    ], style={**section_style, "marginBottom": "18px"})

    # Best policies section (professional cards)
    best_policies_section = html.Div([
        html.H2("Best Policies", style={"marginBottom": "10px", "color": "#2c3e50"}),
        html.Div([
            html.Div([
                html.H4("Slotting", style={"color": "#16a085"}),
                html.P(best_slotting, style={"fontWeight": "bold", "fontSize": "18px", "margin": "0"}),
                html.P(f"Avg Picker Time: {best_slotting_val:.2f} min", style={"fontSize": "14px", "color": "#7f8c8d", "margin": "0"})
            ], style={"background": "#e8f8f5", "borderRadius": "8px", "padding": "14px 16px", "textAlign": "center", "marginRight": "18px", "boxShadow": "0 1px 4px rgba(22,160,133,0.07)"}),
            html.Div([
                html.H4("Batching", style={"color": "#e67e22"}),
                html.P(best_batching, style={"fontWeight": "bold", "fontSize": "18px", "margin": "0"}),
                html.P(f"Avg Picker Time: {best_batching_val:.2f} min", style={"fontSize": "14px", "color": "#7f8c8d", "margin": "0"})
            ], style={"background": "#fbeee6", "borderRadius": "8px", "padding": "14px 16px", "textAlign": "center", "marginRight": "18px", "boxShadow": "0 1px 4px rgba(230,126,34,0.07)"}),
            html.Div([
                html.H4("Routing", style={"color": "#2980b9"}),
                html.P(best_routing, style={"fontWeight": "bold", "fontSize": "18px", "margin": "0"}),
                html.P(f"Avg Picker Time: {best_routing_val:.2f} min", style={"fontSize": "14px", "color": "#7f8c8d", "margin": "0"})
            ], style={"background": "#eaf2fb", "borderRadius": "8px", "padding": "14px 16px", "textAlign": "center", "boxShadow": "0 1px 4px rgba(41,128,185,0.07)"}),
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "0"})
    ], style={**section_style, "marginBottom": "18px"})

    # Top combos table
    def combos_table(df_top):
        rows = []
        for _, r in df_top.iterrows():
            rows.append(html.Tr([
                html.Td(r["Slotting"]), html.Td(r["Batching"]), html.Td(r["Routing"]),
                html.Td(f"{r['Average Picker Time (min)']:.2f}", style=td_right)
            ]))
        header = html.Tr([
            html.Th("Slotting", style=th_style), html.Th("Batching", style=th_style),
            html.Th("Routing", style=th_style), html.Th("Avg Picker Time (min)", style={**th_style, "textAlign": "right"})
        ])
        return html.Table([html.Thead(header), html.Tbody(rows)], style=table_style)

    top_combos_section = html.Div([
        html.H2("Top 3 Policy Combinations", style={"marginBottom": "8px", "color": "#2c3e50"}),
        combos_table(top_combos)
    ], style=section_style)

    # What-if cards vs a standard baseline (visual-first, emphasized)
    baseline = {"Slotting": "RandomSlotting", "Batching": "RandomBatching", "Routing": "SShapeRouting"}
    baseline_mask = (
        (df["Slotting"] == baseline["Slotting"]) &
        (df["Batching"] == baseline["Batching"]) &
        (df["Routing"] == baseline["Routing"]) 
    )
    baseline_time = df.loc[baseline_mask, "Average Picker Time (min)"].mean()

    def pct_gain(new, base):
        if pd.isna(base) or base == 0 or pd.isna(new):
            return None
        return (base - new) / base * 100.0

    # Best overall combo
    best_overall = combo_group.nsmallest(1, "Average Picker Time (min)").iloc[0]
    best_combo_time = float(best_overall["Average Picker Time (min)"])
    best_combo_gain = pct_gain(best_combo_time, baseline_time)

    # Best per policy with others fixed at baseline
    def best_given(dim, fixed_cols):
        mask = pd.Series(True, index=df.index)
        for c, v in fixed_cols.items():
            mask &= (df[c] == v)
        if not mask.any():
            return None, None
        g = df.loc[mask].groupby(dim)["Average Picker Time (min)"].mean()
        if g.empty:
            return None, None
        name = g.idxmin(); val = float(g.min())
        return (name, val)

    slot_name_bp, slot_time_bp = best_given("Slotting", {"Batching": baseline["Batching"], "Routing": baseline["Routing"]})
    batch_name_bp, batch_time_bp = best_given("Batching", {"Slotting": baseline["Slotting"], "Routing": baseline["Routing"]})
    route_name_bp, route_time_bp = best_given("Routing", {"Slotting": baseline["Slotting"], "Batching": baseline["Batching"]})

    def chip_pct(delta):
        if delta is None or pd.isna(delta):
            return html.Span("n/a", style={"background": "#f0f0f0", "color": "#666", "padding": "2px 8px", "borderRadius": "999px", "fontSize": "12px"})
        good = delta >= 0
        bg = "#e8f8f0" if good else "#fdecea"
        fg = "#1e8449" if good else "#c0392b"
        label = "faster" if good else "slower"
        return html.Span(f"{delta:.1f}% {label}", style={"background": bg, "color": fg, "padding": "2px 8px", "borderRadius": "999px", "fontWeight": 600, "fontSize": "12px"})

    def hero_card(title, combo_txt, gain_pct, new_time, base_time):
        good = (gain_pct is not None and gain_pct >= 0)
        grad = "linear-gradient(135deg, #e8f8f5 0%, #ffffff 60%)" if good else "linear-gradient(135deg, #fdecea 0%, #ffffff 60%)"
        accent = "#16a085" if good else "#c0392b"
        return html.Div([
            html.Div("🏆 " + title, style={"fontSize": "16px", "fontWeight": 700, "color": accent, "marginBottom": "6px"}),
            html.Div([
                html.Div([
                    html.Div(f"{gain_pct:.1f}%" if gain_pct is not None else "n/a", style={"fontSize": "32px", "fontWeight": 800, "lineHeight": 1, "color": accent}),
                    html.Div("faster" if (gain_pct is not None and gain_pct >= 0) else ("slower" if gain_pct is not None else ""), style={"fontSize": "12px", "color": "#666"})
                ], style={"minWidth": "100px"}),
                html.Div([
                    html.Div(combo_txt, style={"fontWeight": 700, "marginBottom": "4px", "color": "#2c3e50"}),
                    html.Div(f"Avg time: {new_time:.2f} min vs {base_time:.2f} min baseline" if (new_time is not None and base_time is not None and not pd.isna(new_time) and not pd.isna(base_time)) else "Avg time vs baseline unavailable", style={"fontSize": "12px", "color": "#555"}),
                ])
            ], style={"display": "flex", "alignItems": "center", "gap": "16px"})
        ], style={
            "flexBasis": "100%", "background": grad, "borderLeft": f"4px solid {accent}", "border": "1px solid #e8e8e8",
            "borderRadius": "10px", "padding": "14px 16px", "boxShadow": "0 2px 8px rgba(0,0,0,0.06)"
        })

    def small_card(icon, title, policy, gain_pct, accent):
        return html.Div([
            html.Div([html.Span(icon, style={"marginRight": "6px"}), html.Span(title)], style={"fontWeight": 700, "color": accent, "marginBottom": "4px"}),
            html.Div(policy if policy else "-", style={"fontWeight": 600, "color": "#2c3e50", "marginBottom": "6px"}),
            chip_pct(gain_pct)
        ], style={
            "background": "#ffffff", "border": "1px solid #ececec", "borderLeft": f"4px solid {accent}",
            "borderRadius": "10px", "padding": "12px 14px", "boxShadow": "0 1px 4px rgba(0,0,0,0.05)", "minWidth": "220px"
        })

    what_if_cards = []
    if not pd.isna(baseline_time):
        # Hero: Best Combo vs Baseline
        what_if_cards.append(
            hero_card(
                "Best Combo vs Baseline",
                f"{best_overall['Slotting']} + {best_overall['Batching']} + {best_overall['Routing']}",
                best_combo_gain,
                best_combo_time,
                float(baseline_time) if not pd.isna(baseline_time) else None,
            )
        )

        # Secondary: Change-one-dimension cards
        if slot_name_bp is not None:
            g = pct_gain(slot_time_bp, baseline_time)
            what_if_cards.append(small_card("📦", "Change Slotting Only", slot_name_bp, g, "#16a085"))
        if batch_name_bp is not None:
            g = pct_gain(batch_time_bp, baseline_time)
            what_if_cards.append(small_card("🧩", "Change Batching Only", batch_name_bp, g, "#e67e22"))
        if route_name_bp is not None:
            g = pct_gain(route_time_bp, baseline_time)
            what_if_cards.append(small_card("🧭", "Change Routing Only", route_name_bp, g, "#2980b9"))
    else:
        # Baseline not available in data message card
        what_if_cards.append(html.Div([
            html.Div("ℹ️ Baseline combo not found in results", style={"fontWeight": 700, "color": "#2c3e50", "marginBottom": "4px"}),
            html.Div(f"Expected baseline: {baseline['Slotting']} + {baseline['Batching']} + {baseline['Routing']}", style={"fontSize": "12px", "color": "#555"}),
        ], style={"background": "#fffef5", "border": "1px solid #f5e6a7", "borderLeft": "4px solid #f1c40f", "borderRadius": "8px", "padding": "12px 14px", "boxShadow": "0 1px 4px rgba(0,0,0,0.05)"}))

    # Comparative table
    comp_rows = []
    def add_row(label, slot, batch, route, t):
        if pd.isna(t):
            delta = None; pct = None
        else:
            delta = None if pd.isna(baseline_time) else (t - baseline_time)
            pct = None if pd.isna(baseline_time) else pct_gain(t, baseline_time)
        comp_rows.append({
            "Label": label, "Slotting": slot, "Batching": batch, "Routing": route,
            "Avg Time (min)": t if not pd.isna(t) else None,
            "Δ Time vs Base (min)": delta, "Δ % vs Base": pct
        })

    add_row("Baseline", baseline["Slotting"], baseline["Batching"], baseline["Routing"], baseline_time)
    add_row("Best Combo", best_overall["Slotting"], best_overall["Batching"], best_overall["Routing"], best_combo_time)
    for _, r in top_combos.iterrows():
        add_row("Top Combo", r["Slotting"], r["Batching"], r["Routing"], float(r["Average Picker Time (min)"]))

    def fmt(v, digits=2, suffix=""):
        if v is None or pd.isna(v):
            return "-"
        return f"{v:.{digits}f}{suffix}"

    table_header = html.Tr([
        html.Th("Label", style=th_style), html.Th("Slotting", style=th_style), html.Th("Batching", style=th_style), html.Th("Routing", style=th_style),
        html.Th("Avg Time (min)", style={**th_style, "textAlign": "right"}),
        html.Th("Δ Time (min)", style={**th_style, "textAlign": "right"}),
        html.Th("Δ %", style={**th_style, "textAlign": "right"}),
    ])
    table_rows = []
    for r in comp_rows:
        table_rows.append(html.Tr([
            html.Td(r["Label"]), html.Td(r["Slotting"]), html.Td(r["Batching"]), html.Td(r["Routing"]),
            html.Td(fmt(r["Avg Time (min)"]), style=td_right),
            html.Td(fmt(r["Δ Time vs Base (min)"]), style=td_right),
            html.Td(fmt(r["Δ % vs Base"], 1, "%"), style=td_right),
        ]))
    comparative_table = html.Div([
        html.H2("Comparative Summary", style={"marginBottom": "8px", "color": "#2c3e50"}),
        html.Table([html.Thead(table_header), html.Tbody(table_rows)], style=table_style)
    ], style=section_style)

    # Plots
    # Slotting
    slotting_barplot = px.bar(
        df.groupby("Slotting", as_index=False)["Average Picker Time (min)"].mean(),
        x="Slotting", y="Average Picker Time (min)", title="Avg Picker Time by Slotting (lower is better)", color="Slotting"
    )
    slotting_time_boxplot = px.box(df, x="Slotting", y="Average Picker Time (min)", color="Slotting", title="Picker Time Distribution by Slotting")
    if "Average Picker Distance (m)" in df:
        slotting_distance_barplot = px.bar(
            df.groupby("Slotting", as_index=False)["Average Picker Distance (m)"].mean(),
            x="Slotting", y="Average Picker Distance (m)", title="Avg Picker Distance by Slotting", color="Slotting"
        )
        slotting_distance_boxplot = px.box(df, x="Slotting", y="Average Picker Distance (m)", color="Slotting", title="Picker Distance Distribution by Slotting")
        slotting_scatterplot = px.scatter(df, x="Average Picker Distance (m)", y="Average Picker Time (min)", color="Slotting", title="Time vs Distance by Slotting", hover_data=["Batching","Routing"]) 
    else:
        emptydf = pd.DataFrame(columns=["x","y"]) 
        slotting_distance_barplot = px.scatter(emptydf)
        slotting_distance_boxplot = px.scatter(emptydf)
        slotting_scatterplot = px.scatter(emptydf)

    # Batching
    picker_time_barplot = px.bar(
        df.groupby("Batching", as_index=False)["Average Picker Time (min)"].mean(),
        x="Batching", y="Average Picker Time (min)", title="Avg Picker Time by Batching", color="Batching"
    )

    # Routing
    if "Average Picker Distance (m)" in df:
        distance_boxplot = px.box(df, x="Routing", y="Average Picker Distance (m)", color="Routing", title="Picker Distance Distribution by Routing")
        distance_vs_time_scatter = px.scatter(df, x="Average Picker Distance (m)", y="Average Picker Time (min)", color="Routing", symbol="Batching", title="Time vs Distance by Routing & Batching")
    else:
        emptydf = pd.DataFrame(columns=["x","y"]) 
        distance_boxplot = px.scatter(emptydf)
        distance_vs_time_scatter = px.scatter(emptydf)

    # Return all 13 outputs
    return (
        summary_section,
        best_policies_section,
        top_combos_section,
        what_if_cards,
        comparative_table,
        slotting_barplot,
        slotting_time_boxplot,
        slotting_distance_barplot,
        slotting_distance_boxplot,
        slotting_scatterplot,
        picker_time_barplot,
        distance_boxplot,
        distance_vs_time_scatter,
    )
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