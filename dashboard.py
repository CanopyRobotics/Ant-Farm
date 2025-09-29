import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import os

# Find the latest results CSV
results_dir = "runallcombos_results"
csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
csv_files.sort(reverse=True)
csv_path = os.path.join(results_dir, csv_files[0]) if csv_files else None

df = pd.read_csv(csv_path) if csv_path else pd.DataFrame()

# Convert numeric columns
for col in ['Total Distance Walked (m)', 'Average Picker Time (min)']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Warehouse Policy Experiment Dashboard"),
    html.Div([
        html.Label("Slotting Policy"),
        dcc.Dropdown(
            id='slotting-dropdown',
            options=[{'label': s, 'value': s} for s in sorted(df['Slotting'].unique())],
            value=df['Slotting'].unique()[0] if not df.empty else None
        ),
        html.Label("Batching Policy"),
        dcc.Dropdown(
            id='batching-dropdown',
            options=[{'label': b, 'value': b} for b in sorted(df['Batching'].unique())],
            value=df['Batching'].unique()[0] if not df.empty else None
        ),
        html.Label("Routing Policy"),
        dcc.Dropdown(
            id='routing-dropdown',
            options=[{'label': r, 'value': r} for r in sorted(df['Routing'].unique())],
            value=df['Routing'].unique()[0] if not df.empty else None
        ),
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Div([
        dcc.Graph(id='distance-boxplot'),
        dcc.Graph(id='picker-time-barplot'),
        dcc.Graph(id='distance-vs-time-scatter'),
    ], style={'width': '68%', 'display': 'inline-block', 'paddingLeft': '2%'}),
])

@app.callback(
    [Output('distance-boxplot', 'figure'),
     Output('picker-time-barplot', 'figure'),
     Output('distance-vs-time-scatter', 'figure')],
    [Input('slotting-dropdown', 'value'),
     Input('batching-dropdown', 'value'),
     Input('routing-dropdown', 'value')]
)
def update_graphs(slotting, batching, routing):
    filtered = df.copy()
    if slotting:
        filtered = filtered[filtered['Slotting'] == slotting]
    if batching:
        filtered = filtered[filtered['Batching'] == batching]
    if routing:
        filtered = filtered[filtered['Routing'] == routing]

    # Boxplot: Total Distance by Routing Policy
    fig1 = px.box(df, x='Routing', y='Total Distance Walked (m)', title='Total Distance by Routing Policy')

    # Barplot: Average Picker Time by Batching Policy
    avg_times = df.groupby('Batching')['Average Picker Time (min)'].mean().reset_index()
    fig2 = px.bar(avg_times, x='Batching', y='Average Picker Time (min)', title='Avg Picker Time by Batching Policy')

    # Scatterplot: Distance vs Picker Time by Routing
    fig3 = px.scatter(
        df,  # Use the full dataframe, not filtered
        x='Total Distance Walked (m)',
        y='Average Picker Time (min)',
        color='Routing',
        symbol='Routing',
        title='Distance vs Picker Time by Routing',
        hover_data=['Slotting', 'Batching', 'Iteration']
    )
    fig3.update_layout(legend_title_text='Routing Policy')
    fig3.update_traces(marker=dict(size=10), selector=dict(mode='markers'))

    return fig1, fig2, fig3

if __name__ == '__main__':
    app.run(debug=True)