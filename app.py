import os
import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
from utils.preprocess import load_calibrated_model
from dash.exceptions import PreventUpdate

#Load model and data
df, model = load_calibrated_model()

# Normalize column names we expect
# We'll treat 'pred_prob' as the churn probability (if your dataset uses 'Churn_Prob', copy it)
if 'Churn_Prob' in df.columns and 'pred_prob' not in df.columns:
    df['pred_prob'] = df['Churn_Prob']

# Ensure required columns exist and types are reasonable
expected_cols = ['tenure', 'Churn', 'pred_prob', 'Contract', 'PaymentMethod', 'MonthlyCharges']
for c in expected_cols:
    if c not in df.columns:
        if c == 'pred_prob':
            df[c] = 0.0
        elif c == 'Churn':
            df[c] = 0
        elif c == 'tenure':
            df[c] = 0
        elif c == 'MonthlyCharges':
            df[c] = 0.0
        else:
            df[c] = ''

# Create tenure buckets if not present
bins = [0, 6, 12, 24, 48, 72]
labels = ['0-6', '7-12', '13-24', '25-48', '49-72']
if 'tenure_bucket' not in df.columns:
    df['tenure_bucket'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)

#Segmentation logic
def segment_customers(df_local):
    """
    Segments customers into risk tiers and assigns retention actions based on churn probability.
    """
    def risk_tier(prob):
        if prob > 0.7:
            return 'High'
        elif prob > 0.4:
            return 'Medium'
        else:
            return 'Low'

    def retention_action(row):
        prob = float(row.get('pred_prob', 0.0) or 0.0)
        contract = str(row.get('Contract', '')).lower()
        payment = str(row.get('PaymentMethod', '')).lower()
        try:
            monthly = float(row.get('MonthlyCharges', 0) or 0)
        except Exception:
            monthly = 0.0
        try:
            tenure = int(row.get('tenure', 0) or 0)
        except Exception:
            tenure = 0

        actions = []
        if prob > 0.7:
            if 'month-to-month' in contract:
                actions.append("Offer 15% discount for switching to a 1-year plan.")
            if 'electronic check' in payment or 'mailed check' in payment:
                actions.append("Incentivize AutoPay (e.g., $5 credit).")
            if monthly > 80 and tenure < 12:
                actions.append("Offer a 10% bill reduction for the next 3 months.")
            if tenure < 6:
                actions.append("Provide a welcome loyalty package.")
        elif prob > 0.4:
            if tenure < 24:
                actions.append("Recommend a value bundle or a free trial of a new service.")
            else:
                actions.append("Send a satisfaction survey with a targeted offer.")
        else:
            actions.append("Maintain engagement via loyalty communications and newsletters.")

        return " || ".join(actions[:2]) if actions else "No immediate action needed."

    # safe conversions
    df_local['pred_prob'] = pd.to_numeric(df_local['pred_prob'], errors='coerce').fillna(0.0)
    df_local['RiskTier'] = df_local['pred_prob'].apply(risk_tier)
    df_local['RetentionAction'] = df_local.apply(retention_action, axis=1)
    return df_local

# Precompute an initial segmentation to show on first load
_initial_segmented_df = segment_customers(df.copy())

# --- Pre-calculate Home Page Stats ---
total_customers = len(df)
churn_customers = int(df['Churn'].sum())
non_churn_customers = total_customers - churn_customers
churn_pct = (churn_customers / total_customers) * 100 if total_customers > 0 else 0
non_churn_pct = (non_churn_customers / total_customers) * 100 if total_customers > 0 else 0


# ---------------- Initialize Dash ----------------
app = dash.Dash(__name__, suppress_callback_exceptions=True, title="Churn Dashboard")
server = app.server

# ---------------- Layout & Navigation ----------------
app.layout = html.Div([
    dcc.Location(id='url'),
    html.Nav([
        html.H3(" Customer Churn Analytics", className="logo"),
        html.Div([
            dcc.Link(" Home", href='/', className='nav-link'),
            dcc.Link(" Dashboard", href='/dashboard', className='nav-link'),
            dcc.Link(" Customer Segmentation", href='/segment', className='nav-link'),
            dcc.Link(" Retention Panel", href='/retention', className='nav-link'),
        ], className="nav-links")
    ], className="navbar"),
    html.Div(id='page-content')
])

# ---------------- HOME ----------------
home_layout = html.Div([
    html.Div([
        html.H1("Telecom Customer Churn Prediction", className="title"),
        html.P("Our Telecom Churn Prediction and Retention Platform empowers businesses to understand customer behavior through data-driven insights.)", className="subtitle"),
        html.P("Using advanced machine learning and calibrated probability modeling, it identifies customers most likely to churn and suggests personalized retention strategies to keep them engaged.),"),
        html.P("The interactive dashboard enables you to visualize churn trends, evaluate customer risk segments, and simulate pilot retention campaigns — all in one unified analytics workspace."),
    ], className="hero"),

    html.Div([
        html.Div([
            html.H3("CHURNED CUSTOMERS"),
            html.P(f"{churn_customers:,}", className="stat-number"),
            html.P(f"{churn_pct:.1f}% of Total")
        ], className="stat-card", style={'border-top': '4px solid #dc3545'}),

        html.Div([
            html.H3("NON-CHURNED CUSTOMERS"),
            html.P(f"{non_churn_customers:,}", className="stat-number"),
            html.P(f"{non_churn_pct:.1f}% of Total")
        ], className="stat-card", style={'border-top': '4px solid #198754'}),

    ], className="two-cols", style={'marginTop': '2rem'}),

])


# ---------------- DASHBOARD ----------------
dashboard_layout = html.Div([
    html.H1("📊 Churn Analytics Dashboard", className="title"),

    html.Div([
        html.Label("Filter by Churn Status:"),
        dcc.Dropdown(
            id='churn-filter',
            options=[
                {'label': 'All Customers', 'value': 'all'},
                {'label': 'Churned', 'value': 1},
                {'label': 'Non-Churned', 'value': 0},
            ],
            value='all', clearable=False
        ),
    ], className="dropdown"),

    html.Div([
        dcc.Graph(id='bar-contract'),
        dcc.Graph(id='bar-tenure'),
    ], className="two-cols"),

    html.Div([
        html.H3("Calibrated Churn Probability Distribution"),
        dcc.Graph(
            id="prob-dist",
            figure=px.histogram(df, x='pred_prob', nbins=25,
                                title="Churn Probability Distribution",
                                color_discrete_sequence=['#1f77b4'])
        )
    ], className="chart-container"),

    html.Div([
        html.H3("Churn Probability by Contract × Tenure Bucket"),
        dcc.Graph(
            id="heatmap",
            figure=px.density_heatmap(
                df, x='Contract', y='tenure_bucket', z='pred_prob',
                color_continuous_scale='RdYlGn_r',
                title="Predicted Churn Probability by Segment")
        )
    ], className="chart-container"),
])

# ---------------- SEGMENTATION ----------------
default_cols = ['customerID', 'pred_prob', 'RiskTier', 'RetentionAction', 'Contract', 'tenure', 'MonthlyCharges', 'PaymentMethod']
present_cols = [c for c in default_cols if c in _initial_segmented_df.columns]

summary_init = _initial_segmented_df['RiskTier'].value_counts().reindex(['High', 'Medium', 'Low']).fillna(0).reset_index()
summary_init.columns = ['RiskTier', 'Count']
fig_init = px.bar(summary_init, x='RiskTier', y='Count', title='Segment Counts by Risk Tier')

segment_layout = html.Div([
    html.H1(" Customer Segmentation & Retention Actions", className="title"),

    html.Div([
        html.Div([
            html.Label("Risk Tier Filter:"),
            dcc.Dropdown(
                id='seg-risk-filter',
                options=[
                    {'label': 'All', 'value': 'all'},
                    {'label': 'High', 'value': 'High'},
                    {'label': 'Medium', 'value': 'Medium'},
                    {'label': 'Low', 'value': 'Low'},
                ],
                value='all', clearable=False
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '1rem'}),

        html.Div([
            html.Label("Show top N by predicted probability:"),
            dcc.Input(id='seg-top-n', type='number', value=100, min=1, step=1)
        ], style={'width': '20%', 'display': 'inline-block', 'marginRight': '1rem'}),

        html.Button("Run Segmentation", id='run-seg', n_clicks=0, className='btn'),
        html.Button("Download CSV", id='download-seg', n_clicks=0, className='btn', style={'marginLeft': '1rem'}),
        dcc.Download(id='download-seg-file'),
    ], className="controls", style={'marginBottom': '1rem'}),

    html.Div([
        html.H4("Segment Summary"),
        dcc.Graph(id='seg-summary-fig', figure=fig_init)
    ], className='chart-container'),

    html.Div([
        html.H4("Segmented Customers"),
        dash_table.DataTable(
            id='seg-table',
            columns=[{"name": c, "id": c} for c in present_cols],
            data=_initial_segmented_df[present_cols].sort_values('pred_prob', ascending=False).to_dict('records'),
            page_size=15,
            sort_action='native',
            filter_action='native',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'whiteSpace': 'normal', 'height': 'auto'},
        )
    ], className='table-container')
])

# ---------------- RETENTION SIMULATION PAGE ----------------
# default file path for saved outputs
OUTPUT_DIR = "output_playbooks"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "telco_with_next_steps.csv")

retention_layout = html.Div([
    html.H1(" Retention - Pilot Simulation & Next Steps", className="title"),

    html.Div([
        html.Div([
            html.Label("Pilot offer percent for High-risk (0-100):"),
            dcc.Input(id='pilot-pct', type='number', value=20, min=0, max=100, step=1)
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '1rem'}),

        html.Div([
            html.Label("Assumed success rate for offers (0-100):"),
            dcc.Input(id='success-pct', type='number', value=55, min=0, max=100, step=1)
        ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '1rem'}),

        html.Div([
            html.Label("Random seed (optional):"),
            dcc.Input(id='sim-seed', type='number', value=42)
        ], style={'width': '20%', 'display': 'inline-block'}),
    ], style={'marginBottom': '1rem'}),

    html.Div([
        html.Button("Run Retention Simulation", id='run-retention', n_clicks=0, className='btn'),
        html.Button("Download Results CSV", id='download-retention', n_clicks=0, className='btn', style={'marginLeft': '1rem'}),
        dcc.Download(id='download-retention-file'),
    ], style={'marginBottom': '1rem'}),

    html.Div([
        html.H4("Pilot Offer Distribution by Risk Tier"),
        dcc.Graph(id='offer-dist-fig')
    ], className='chart-container'),

    html.Div([
        html.H4("Retention Outcome for Offered Customers"),
        dcc.Graph(id='retention-outcome-fig')
    ], className='chart-container'),

    html.Div([
        html.H4("Churn Rate Before vs After"),
        dcc.Graph(id='before-after-fig')
    ], className='chart-container'),

    html.Div([
        html.H4("Top Retention Actions by Success Rate"),
        dcc.Graph(id='action-success-fig')
    ], className='chart-container'),

    html.Div([
        html.H4("Retention Monitoring Summary"),
        html.Ul(id='retention-summary')
    ], className='chart-container')
])

# ---------------- CALLBACKS ----------------
@app.callback(
    [Output('bar-contract', 'figure'),
     Output('bar-tenure', 'figure')],
    [Input('churn-filter', 'value')]
)
def update_dashboard(filter_value):
    dff = df.copy()
    if filter_value != 'all':
        dff = dff[dff['Churn'] == int(filter_value)]

    contract_fig = px.bar(
        dff.groupby('Contract')['Churn'].mean().reset_index(),
        x='Contract', y='Churn', labels={'Churn': 'Churn Rate'},
        color='Contract', title='Churn Rate by Contract Type'
    )

    tenure_fig = px.bar(
        dff.groupby('tenure_bucket')['Churn'].mean().reset_index(),
        x='tenure_bucket', y='Churn', labels={'Churn': 'Churn Rate'},
        color='tenure_bucket', title='Churn Rate by Tenure Bucket'
    )

    return contract_fig, tenure_fig

# ---- Segmentation callback (filter FIRST, then sort & top-N) ----
@app.callback(
    Output('seg-table', 'data'),
    Output('seg-summary-fig', 'figure'),
    Input('run-seg', 'n_clicks'),
    State('seg-risk-filter', 'value'),
    State('seg-top-n', 'value'),
    prevent_initial_call=True
)
def run_segmentation(n_clicks, risk_filter, top_n):
    seg_df = segment_customers(df.copy())

    # 1) Apply risk-tier filter FIRST
    if risk_filter and risk_filter != 'all':
        seg_df = seg_df[seg_df['RiskTier'] == risk_filter]

    # 2) Parse top_n safely
    try:
        top_n_val = int(top_n) if top_n not in (None, '', 0) else None
        if top_n_val is not None and top_n_val <= 0:
            top_n_val = None
    except Exception:
        top_n_val = None

    # 3) Sort and optionally slice
    seg_df = seg_df.sort_values('pred_prob', ascending=False)
    if top_n_val is not None:
        seg_df = seg_df.head(top_n_val)

    # 4) Build summary figure
    summary = seg_df['RiskTier'].value_counts().reindex(['High', 'Medium', 'Low']).fillna(0).reset_index()
    summary.columns = ['RiskTier', 'Count']
    fig = px.bar(summary, x='RiskTier', y='Count', title='Segment Counts by Risk Tier')

    # 5) Save latest for download
    global _latest_segmented_df
    _latest_segmented_df = seg_df

    # 6) Present only available columns in the table
    present = [c for c in default_cols if c in seg_df.columns]
    data = seg_df[present].to_dict('records')
    return data, fig

@app.callback(
    Output('download-seg-file', 'data'),
    Input('download-seg', 'n_clicks'),
    prevent_initial_call=True
)
def download_segment_csv(n_clicks):
    global _latest_segmented_df
    if '_latest_segmented_df' in globals() and _latest_segmented_df is not None:
        to_download = _latest_segmented_df
    else:
        to_download = _initial_segmented_df

    if to_download is None or to_download.empty:
        raise PreventUpdate

    return dcc.send_data_frame(to_download.to_csv, "segmented_customers.csv", index=False)

# ---------------- Retention simulation logic (based on rate.py) ----------------
def run_retention_simulation(df_input, pilot_pct=20, success_pct=55, seed=42):
    """
    Simulate pilot offers and retention outcomes.
    - pilot_pct: percent of High-risk customers to receive offers (0-100)
    - success_pct: percent of offered customers who are retained (0-100)
    Returns: DataFrame (copy) with OfferGiven, Retained_After_Offer, Updated_Churn_Label, Scaled_Action (placeholder)
    """
    df_sim = df_input.copy()
    df_sim['pred_prob'] = pd.to_numeric(df_sim['pred_prob'], errors='coerce').fillna(0.0)
    # Ensure RiskTier exists
    if 'RiskTier' not in df_sim.columns:
        df_sim = segment_customers(df_sim)

    rng = np.random.RandomState(seed if seed is not None else None)

    # Pilot offer: only consider High risk
    pct = float(pilot_pct) / 100.0 if pilot_pct is not None else 0.0
    df_sim['OfferGiven'] = np.where(
        (df_sim['RiskTier'] == 'High') & (rng.rand(len(df_sim)) < pct),
        'Yes', 'No'
    )

    # Retained after offer: offered customers retained with probability success_pct
    success_prob = float(success_pct) / 100.0 if success_pct is not None else 0.0
    df_sim['Retained_After_Offer'] = np.where(
        (df_sim['OfferGiven'] == 'Yes') & (rng.rand(len(df_sim)) < success_prob),
        'Yes', 'No'
    )

    # For customers without offers, follow rate.py rule:
    # if Churn_Prob (pred_prob) > 0.7 -> Not retained, else retained
    df_sim['Retained_After_Offer'] = np.where(
        (df_sim['OfferGiven'] == 'No'),
        np.where(df_sim['pred_prob'] > 0.7, 'No', 'Yes'),
        df_sim['Retained_After_Offer']
    )

    # Updated churn label: 0 if retained, 1 if churned
    df_sim['Updated_Churn_Label'] = np.where(df_sim['Retained_After_Offer'] == 'Yes', 0, 1)

    # Compute before and after churn rates
    before_churn_rate = (df_sim['pred_prob'] > 0.5).mean() * 100.0
    after_churn_rate = df_sim['Updated_Churn_Label'].mean() * 100.0

    # Evaluate retention success among offered customers
    offer_df = df_sim[df_sim['OfferGiven'] == 'Yes']
    if not offer_df.empty and 'RetentionAction' in offer_df.columns:
        action_success = offer_df.groupby('RetentionAction')['Retained_After_Offer'].apply(
            lambda x: (x == 'Yes').mean()
        ).sort_values(ascending=False)
    else:
        action_success = pd.Series(dtype=float)

    # Scale top action to remaining high-risk customers (simulate)
    top_action = action_success.index[0] if len(action_success) > 0 else None
    if top_action:
        df_sim['Scaled_Action'] = np.where(
            (df_sim['RiskTier'] == 'High') & (df_sim['OfferGiven'] == 'No'),
            top_action, df_sim.get('RetentionAction', '')
        )
    else:
        df_sim['Scaled_Action'] = df_sim.get('RetentionAction', '')

    # Monitoring summary
    summary = {
        "Total Customers": len(df_sim),
        "High Risk Customers": int((df_sim['RiskTier'] == 'High').sum()),
        "Pilot Offers Sent": int((df_sim['OfferGiven'] == 'Yes').sum()),
        "Retention Success Rate (offered)": float(((df_sim.loc[df_sim['OfferGiven']=='Yes','Retained_After_Offer']=='Yes').mean())*100) if (df_sim['OfferGiven']=='Yes').any() else 0.0,
        "Churn Rate Before (%)": float(before_churn_rate),
        "Churn Rate After (%)": float(after_churn_rate),
        "Estimated Reduction (%)": float(before_churn_rate - after_churn_rate)
    }

    return df_sim, action_success, summary

# Retention page callback: run simulation and update UI elements
@app.callback(
    Output('offer-dist-fig', 'figure'),
    Output('retention-outcome-fig', 'figure'),
    Output('before-after-fig', 'figure'),
    Output('action-success-fig', 'figure'),
    Output('retention-summary', 'children'),
    Input('run-retention', 'n_clicks'),
    State('pilot-pct', 'value'),
    State('success-pct', 'value'),
    State('sim-seed', 'value'),
    prevent_initial_call=True
)
def run_retention(n_clicks, pilot_pct, success_pct, sim_seed):
    # Run simulation
    sim_df, action_success, summary = run_retention_simulation(df.copy(), pilot_pct=pilot_pct, success_pct=success_pct, seed=sim_seed)

    # Offer distribution by RiskTier
    offer_counts = sim_df.groupby(['RiskTier', 'OfferGiven']).size().reset_index(name='count')
    offer_fig = px.bar(offer_counts, x='RiskTier', y='count', color='OfferGiven', barmode='group',
                       title='Pilot Offer Distribution by Risk Tier')

    # Retention outcome for offered customers
    offered = sim_df[sim_df['OfferGiven'] == 'Yes']
    if offered.empty:
        retention_out_fig = px.bar(pd.DataFrame({'Outcome': [], 'count': []}), x='Outcome', y='count',
                                   title='Retention Outcome for Pilot Offers (no offers sent)')
    else:
        outcome_counts = offered['Retained_After_Offer'].value_counts().reset_index()
        outcome_counts.columns = ['Outcome', 'count']
        retention_out_fig = px.bar(outcome_counts, x='Outcome', y='count', title='Retention Outcome for Pilot Offers')

    # Before vs After churn rates
    before = summary['Churn Rate Before (%)']
    after = summary['Churn Rate After (%)']
    before_after_fig = px.bar(pd.DataFrame({'Stage': ['Before', 'After'], 'ChurnRate': [before, after]}),
                              x='Stage', y='ChurnRate', title='Churn Rate Before vs After (%)')

    # Top retention actions success
    if not action_success.empty:
        action_df = action_success.reset_index()
        action_df.columns = ['RetentionAction', 'SuccessRate']
        action_fig = px.bar(action_df, x='SuccessRate', y='RetentionAction', orientation='h', title='Retention Action Success Rates')
    else:
        action_fig = px.bar(pd.DataFrame({'RetentionAction': [], 'SuccessRate': []}), x='SuccessRate', y='RetentionAction', orientation='h',
                            title='Retention Action Success Rates (no offers)')

    # Summary list items
    items = []
    for k, v in summary.items():
        items.append(html.Li(f"{k}: {v}"))

    # Save the latest simulated df globally for download
    global _latest_retention_df
    _latest_retention_df = sim_df

    # Also save CSV to disk (as rate.py did)
    try:
        sim_df.to_csv(OUTPUT_CSV, index=False)
    except Exception as e:
        # don't crash app if save fails; just continue
        print("Warning: could not save retention CSV:", e)

    return offer_fig, retention_out_fig, before_after_fig, action_fig, items

@app.callback(
    Output('download-retention-file', 'data'),
    Input('download-retention', 'n_clicks'),
    prevent_initial_call=True
)
def download_retention(n_clicks):
    # Prefer latest computed sim; else fallback to saved file if exists
    global _latest_retention_df
    if '_latest_retention_df' in globals() and _latest_retention_df is not None:
        return dcc.send_data_frame(_latest_retention_df.to_csv, "telco_with_next_steps.csv", index=False)
    elif os.path.exists(OUTPUT_CSV):
        return dcc.send_file(OUTPUT_CSV)
    else:
        raise PreventUpdate

# Page routing
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/dashboard':
        return dashboard_layout
    if pathname == '/segment':
        return segment_layout
    if pathname == '/retention':
        return retention_layout
    return home_layout

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True, port=8050)
