import os
import joblib
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

# ── Load Artifacts ─────────────────────────────────────────────────────────────
model = joblib.load(os.path.join(BASE_DIR, 'artifacts', 'model_xgb.pkl'))
preprocessor = joblib.load(os.path.join(BASE_DIR, 'artifacts', 'preprocessor.pkl'))
kmeans = joblib.load(os.path.join(BASE_DIR, 'artifacts', 'kmeans_model.pkl'))
cluster_scaler = joblib.load(os.path.join(BASE_DIR, 'artifacts', 'cluster_scaler.pkl'))

# ── Constants ──────────────────────────────────────────────────────────────────
# Nathan's model found 2 clusters dynamically — medium risk and low risk
CLUSTER_LABELS = {
    0: 'Medium Risk',
    1: 'Low Risk'
}

CLUSTER_COLORS = {
    0: '#ef4444',
    1: '#22c55e'
}

RISK_LEVELS = {
    'No Default': ('Low Risk', '#22c55e'),
    'Default Risk': ('High Risk', '#ef4444')
}

STAGE_COLORS = {
    'No Default': '#22c55e',
    'Default Risk': '#ef4444'
}

# ── Clustering features in the exact order Nathan used ─────────────────────────
CLUSTER_FEATURES = [
    'person_income',
    'loan_amnt',
    'loan_int_rate',
    'person_emp_length',
    'cb_person_cred_hist_length',
    'person_age'
]

# ── Helper Functions ───────────────────────────────────────────────────────────
def input_field(label, component):
    return html.Div([
        html.Label(label, className='field-label'),
        component
    ], className='field-wrap mb-3')


def get_shap_image(filename):
    # Dash serves files from assets_folder as /assets/filename
    # assets_folder is set to reports/figures so images are served directly
    return f"/assets/{filename}"


# ── App Initialisation ─────────────────────────────────────────────────────────
# Point assets_folder to reports/figures so Dash serves images from there directly
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    assets_folder=os.path.join(BASE_DIR, 'reports', 'figures')
)
server = app.server
app.title = 'Credit Risk Assessment System'

# ── Layout ─────────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # Navbar
    html.Nav([
        html.Div([
            html.Span('MLG | GROUP 2', style={'fontSize': '1.2rem', 'marginRight': '0.75rem', 'color': 'white'}),
            html.Span('Credit Risk Assessment System', style={'fontWeight': 'bold', 'color': 'white', 'fontSize': '1.1rem'}),
        ], style={'display': 'flex', 'alignItems': 'center'}),
    ], style={
        'backgroundColor': '#1E3A5F',
        'padding': '15px 30px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'space-between'
    }),

    # Page body
    html.Div([

        # Hero
        html.Div([
            html.H1('Credit Risk Assessment System', style={
                'textAlign': 'center', 'color': '#1E3A5F',
                'fontWeight': 'bold', 'marginBottom': '10px'
            }),
            html.P(
                'Real-time default probability engine powered by XGBoost and K-Means clustering.',
                style={'textAlign': 'center', 'color': '#555', 'marginBottom': '30px'}
            )
        ], style={'padding': '30px 0 10px 0'}),

        # Input card
        html.Div([
            html.H2('Applicant Profile', style={'color': '#1E3A5F', 'marginBottom': '5px'}),
            html.P('Enter financial and demographic details to run a credit risk assessment.',
                   style={'color': '#777', 'marginBottom': '20px'}),

            # Demographics
            html.P('Demographics', style={'fontWeight': 'bold', 'color': '#2E6DA4', 'marginBottom': '10px'}),
            dbc.Row([
                dbc.Col(input_field("Age (Years)",
                    dbc.Input(id='person_age', type='number', value=35)), width=4),
                dbc.Col(input_field("Employment Length (Years)",
                    dbc.Input(id='person_emp_length', type='number', value=3)), width=4),
                dbc.Col(input_field("Credit History Length (Years)",
                    dbc.Input(id='cb_person_cred_hist_length', type='number', value=5)), width=4),
            ]),

            # Financials
            html.P('Financials', style={'fontWeight': 'bold', 'color': '#2E6DA4',
                                        'marginTop': '15px', 'marginBottom': '10px'}),
            dbc.Row([
                dbc.Col(input_field("Annual Income ($)",
                    dbc.Input(id='person_income', type='number', value=65000)), width=4),
                dbc.Col(input_field("Previous Default on File",
                    dcc.Dropdown(
                        id='cb_person_default_on_file',
                        options=[{'label': 'Yes', 'value': 'Y'}, {'label': 'No', 'value': 'N'}],
                        value='N', clearable=False
                    )), width=4),
                dbc.Col(input_field("Loan % of Income",
                    dbc.Input(id='loan_percent_income', type='number', value=0.23, step=0.01)), width=4),
            ]),

            # Loan Details
            html.P('Loan Details', style={'fontWeight': 'bold', 'color': '#2E6DA4',
                                           'marginTop': '15px', 'marginBottom': '10px'}),
            dbc.Row([
                dbc.Col(input_field("Loan Amount ($)",
                    dbc.Input(id='loan_amnt', type='number', value=15000)), width=4),
                dbc.Col(input_field("Interest Rate (%)",
                    dbc.Input(id='loan_int_rate', type='number', value=10.5, step=0.1)), width=4),
                dbc.Col(input_field("Loan Intent",
                    dcc.Dropdown(
                        id='loan_intent',
                        options=[
                            {'label': 'Personal', 'value': 'PERSONAL'},
                            {'label': 'Education', 'value': 'EDUCATION'},
                            {'label': 'Home Improvement', 'value': 'HOMEIMPROVEMENT'},
                            {'label': 'Medical', 'value': 'MEDICAL'},
                            {'label': 'Venture', 'value': 'VENTURE'},
                            {'label': 'Debt Consolidation', 'value': 'DEBTCONSOLIDATION'}
                        ],
                        value='PERSONAL', clearable=False
                    )), width=4),
            ]),
            dbc.Row([
                dbc.Col(input_field("Home Ownership",
                    dcc.Dropdown(
                        id='person_home_ownership',
                        options=[
                            {'label': 'Rent', 'value': 'RENT'},
                            {'label': 'Mortgage', 'value': 'MORTGAGE'},
                            {'label': 'Own', 'value': 'OWN'},
                            {'label': 'Other', 'value': 'OTHER'}
                        ],
                        value='MORTGAGE', clearable=False
                    )), width=6),
            ]),

            # Submit button
            html.Div([
                dbc.Button("RUN RISK ASSESSMENT", id='predict-btn',
                           color='primary', size='lg',
                           style={'width': '100%', 'marginTop': '20px',
                                  'backgroundColor': '#1E3A5F', 'border': 'none'})
            ])

        ], style={
            'backgroundColor': 'white',
            'padding': '30px',
            'borderRadius': '12px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
            'marginBottom': '30px'
        }),

        # Results
        html.Div(id='results-output')

    ], style={'maxWidth': '1100px', 'margin': '0 auto', 'padding': '0 20px 40px 20px'})
])


# ── Callback ───────────────────────────────────────────────────────────────────
@app.callback(
    Output('results-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('person_age', 'value'),
    State('person_income', 'value'),
    State('loan_amnt', 'value'),
    State('person_home_ownership', 'value'),
    State('loan_int_rate', 'value'),
    State('loan_intent', 'value'),
    State('person_emp_length', 'value'),
    State('cb_person_cred_hist_length', 'value'),
    State('cb_person_default_on_file', 'value'),
    State('loan_percent_income', 'value'),
    prevent_initial_call=True
)
def run_assessment(n_clicks, person_age, person_income, loan_amnt,
                   person_home_ownership, loan_int_rate, loan_intent,
                   person_emp_length, cb_person_cred_hist_length,
                   cb_person_default_on_file, loan_percent_income):

    preprocessor_columns = [
        'person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
        'loan_intent', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_default_on_file', 'cb_person_cred_hist_length'
    ]

    applicant = {
        'person_age': int(person_age) if person_age else 35,
        'person_income': float(person_income) if person_income else 65000,
        'person_home_ownership': str(person_home_ownership),
        'person_emp_length': float(person_emp_length) if person_emp_length else 3,
        'loan_intent': str(loan_intent),
        'loan_amnt': float(loan_amnt) if loan_amnt else 15000,
        'loan_int_rate': float(loan_int_rate) if loan_int_rate else 10.5,
        'loan_percent_income': float(loan_percent_income) if loan_percent_income else 0.23,
        'cb_person_default_on_file': str(cb_person_default_on_file) if cb_person_default_on_file else 'N',
        'cb_person_cred_hist_length': float(cb_person_cred_hist_length) if cb_person_cred_hist_length else 5,
    }

    try:
        # ── Classification ─────────────────────────────────────────────────────
        X_input = pd.DataFrame([applicant])[preprocessor_columns]
        X_processed = preprocessor.transform(X_input)

        pred_proba = model.predict_proba(X_processed)[0]
        pred_idx = int(model.predict(X_processed)[0])
        pred_label = 'Default Risk' if pred_idx == 1 else 'No Default'
        risk_label, risk_color = RISK_LEVELS.get(pred_label, ('Unknown', '#2563eb'))

        # ── Cluster Assignment ─────────────────────────────────────────────────
        cluster_input = pd.DataFrame([{
            'person_income': applicant['person_income'],
            'loan_amnt': applicant['loan_amnt'],
            'loan_int_rate': applicant['loan_int_rate'],
            'person_emp_length': applicant['person_emp_length'],
            'cb_person_cred_hist_length': applicant['cb_person_cred_hist_length'],
            'person_age': applicant['person_age']
        }])[CLUSTER_FEATURES]

        cluster_scaled = cluster_scaler.transform(cluster_input)
        cluster_id = int(kmeans.predict(cluster_scaled)[0])
        cluster_name = CLUSTER_LABELS.get(cluster_id, 'Unknown')
        cluster_color = CLUSTER_COLORS.get(cluster_id, '#2563eb')

        # ── Recommendation ─────────────────────────────────────────────────────
        if pred_label == 'Default Risk':
            rec_text = (
                "CAUTION: High risk of default detected. Recommend further manual review. "
                "Key risk factors include high loan-to-income ratio and elevated interest rate. "
                "Consider requesting additional financial documentation before approval."
            )
        else:
            rec_text = (
                f"APPROVAL RECOMMENDED: Stable applicant profile within the '{cluster_name}' segment. "
                f"Standard credit checks apply. Monitor loan-to-income ratio for ongoing risk management."
            )

        # ── Probability Chart ──────────────────────────────────────────────────
        proba_fig = go.Figure(go.Bar(
            x=['No Default', 'Default Risk'],
            y=[pred_proba[0], pred_proba[1]],
            marker_color=[STAGE_COLORS['No Default'], STAGE_COLORS['Default Risk']],
            text=[f'{p*100:.1f}%' for p in pred_proba],
            textposition='auto',
        ))
        proba_fig.update_layout(
            title='Prediction Probabilities',
            yaxis=dict(range=[0, 1]),
            template='plotly_white',
            height=300,
            margin=dict(t=40, b=20, l=40, r=20)
        )

        # ── Feature Importance Chart ───────────────────────────────────────────
        importances = model.feature_importances_
        feature_names = preprocessor.get_feature_names_out().tolist()
        clean_names = [
            n.replace("num__", "").replace("cat__", "").replace("_", " ")
            for n in feature_names
        ]

        indices = np.argsort(importances)[::-1][:10]
        top_features = [clean_names[i] for i in indices if i < len(clean_names)]
        top_vals = [importances[i] for i in indices if i < len(importances)]

        importance_fig = go.Figure(go.Bar(
            x=top_vals[::-1],
            y=top_features[::-1],
            orientation='h',
            marker_color='#1E3A5F'
        ))
        importance_fig.update_layout(
            title='Top 10 Feature Drivers',
            template='plotly_white',
            height=350,
            margin=dict(t=40, b=20, l=180, r=20)
        )

        # ── SHAP Image Helper ──────────────────────────────────────────────────
        def shap_img(filename, title):
            return dbc.Col([
                html.P(title, style={'fontWeight': 'bold', 'marginTop': '10px', 'color': '#1E3A5F'}),
                html.Img(
                    src=get_shap_image(filename),
                    style={'width': '100%', 'borderRadius': '8px', 'border': '1px solid #ddd'}
                )
            ], md=6)

        # ── Results Layout ─────────────────────────────────────────────────────
        return html.Div([

            html.H3('Assessment Report', style={'color': '#1E3A5F', 'marginBottom': '20px'}),

            # Stat cards
            html.Div([
                html.Div([
                    html.P('Prediction', style={'color': '#777', 'marginBottom': '5px'}),
                    html.H4(pred_label, style={'color': risk_color, 'fontWeight': 'bold'})
                ], style={'flex': '1', 'padding': '20px', 'borderLeft': f'5px solid {risk_color}',
                           'background': '#f9f9f9', 'margin': '5px', 'borderRadius': '8px'}),

                html.Div([
                    html.P('Confidence', style={'color': '#777', 'marginBottom': '5px'}),
                    html.H4(f"{pred_proba[pred_idx]*100:.1f}%",
                            style={'color': '#1E3A5F', 'fontWeight': 'bold'})
                ], style={'flex': '1', 'padding': '20px', 'borderLeft': '5px solid #1E3A5F',
                           'background': '#f9f9f9', 'margin': '5px', 'borderRadius': '8px'}),

                html.Div([
                    html.P('Borrower Segment', style={'color': '#777', 'marginBottom': '5px'}),
                    html.H4(cluster_name, style={'color': cluster_color, 'fontWeight': 'bold'})
                ], style={'flex': '1', 'padding': '20px', 'borderLeft': f'5px solid {cluster_color}',
                           'background': '#f9f9f9', 'margin': '5px', 'borderRadius': '8px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px'}),

            # Recommendation
            html.Div([
                html.P('Recommendation', style={'fontWeight': 'bold', 'color': '#1E3A5F', 'marginBottom': '5px'}),
                html.P(rec_text, style={'color': '#444'})
            ], style={'background': '#EEF6FF', 'padding': '20px',
                      'borderRadius': '10px', 'marginBottom': '20px'}),

            # Charts
            dbc.Row([
                dbc.Col(dcc.Graph(figure=proba_fig), md=6),
                dbc.Col(dcc.Graph(figure=importance_fig), md=6)
            ], className='mb-4'),

            # SHAP Analysis
            html.Div([
                html.H4('SHAP Analysis — Key Risk Drivers',
                        style={'color': '#1E3A5F', 'marginBottom': '15px',
                               'borderTop': '1px solid #eee', 'paddingTop': '20px'}),
                html.P(
                    'SHAP values show which features most strongly influence the model '
                    'predictions globally and for individual applicants.',
                    style={'color': '#555', 'marginBottom': '20px'}
                ),
                dbc.Row([
                    shap_img('shap_bar.png', 'Global Feature Importance (Bar)'),
                    shap_img('shap_beeswarm.png', 'Global Feature Importance (Beeswarm)'),
                ]),
                dbc.Row([
                    shap_img('shap_waterfall.png', 'Local Explanation — High Risk Applicant'),
                    shap_img('cluster_pca.png', 'Borrower Cluster Visualisation (PCA)'),
                ], className='mt-3'),
            ])

        ], style={
            'padding': '30px',
            'background': 'white',
            'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
            'borderRadius': '12px'
        })

    except Exception as e:
        import traceback
        return html.Div([
            html.H4('Assessment Error', style={'color': '#ef4444'}),
            html.Pre(traceback.format_exc(),
                     style={'background': '#f8f8f8', 'padding': '15px',
                            'fontSize': '12px', 'color': 'red', 'borderRadius': '8px'})
        ])


if __name__ == '__main__':
    app.run(debug=True)