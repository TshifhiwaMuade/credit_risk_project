import os
import joblib
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# Setup path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
SHAP_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

# Loading artifacts
model = joblib.load(os.path.join(BASE_DIR, 'artifacts', 'model_xgb.pkl'))
preprocessor = joblib.load(os.path.join(BASE_DIR, 'artifacts', 'preprocessor.pkl'))
kmeans = joblib.load(os.path.join(BASE_DIR, 'artifacts', 'kmeans_model.pkl'))
cluster_scaler = joblib.load(os.path.join(BASE_DIR, 'artifacts', 'cluster_scaler.pkl'))

# UI constants
CLUSTER_LABELS = {
    0: 'Conservative Borrowers', 
    1: 'High-Leverage Risk Group', 
    2: 'Stable Mid-Tier'
}
STATUS_COLORS = {
    'No Default': '#58427C',  
    'Default Risk': '#800080'  
}
RISK_LEVELS = {
    'No Default': ('Low Risk', '#22c55e'),
    'Default Risk': ('High Risk', '#ef4444')
}
STAGE_COLORS = {
    'No Default': '#58427C',
    'Default Risk': '#800080'
}

def input_field(label, component):
    return html.Div([
        html.Label(label, className='field-label'),
        component
    ], className='field-wrap mb-3')

# App Initialization
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = 'Credit Risk Assessment System'

# Layout
app.layout = html.Div([
    html.Nav([
        html.Div([
            html.Span('MLG | GROUP 2', style={'fontSize': '1.8rem', 'marginRight': '0.75rem', 'color': 'white'}),
            html.Span('Credit Risk Assessment System', className='brand-bold'),
        ], className='flex items-center'),
        html.Span('', className='badge', style={'backgroundColor': 'var(--rich-mauve)', 'color': 'var(--cyber-grape)'})
    ], className='app-nav'),

    html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1(['Credit Risk', html.Br(), 'Prediction System'], className='hero-title text-center'),
                    html.P("Real-time default probability engine powered by XGBoost and K-Means clustering.", className="text-lg opacity-80 text-center")
                ], className='p-12 text-center')
            ], lg=6, md=12, className='d-flex justify-content-center'),

            dbc.Col([
                html.Div([
                    html.H2('Applicant Profile', className='card-header-title text-2xl mb-4'),
                    html.P('Financial and demographic metrics.', className='text-sm opacity-60 mb-4'),

                    html.Div([
                        html.Div([
                            html.Span('👤', style={'fontSize': '1.2rem', 'marginRight': '0.5rem'}),
                            html.Span('Demographics', className='section-title')
                        ], className='section-header mb-3'),

                        dbc.Row([
                            dbc.Col(input_field("Age (Years)", dbc.Input(id='person_age', type='number', value=35, className='form-control-modern')), width=4),
                            dbc.Col(input_field("Employment Length (Years)", dbc.Input(id='person_emp_length', type='number', value=3, className='form-control-modern')), width=4),
                            dbc.Col(input_field("Credit History Length (Years)", dbc.Input(id='cb_person_cred_hist_length', type='number', value=5, className='form-control-modern')), width=4),
                        ])
                    ], className='section-card mb-4'),

                    html.Div([
                        html.Div([
                            html.Span('📈', style={'fontSize': '1.2rem', 'marginRight': '0.5rem'}),
                            html.Span('Financials', className='section-title')
                        ], className='section-header mb-3'),

                        dbc.Row([
                            dbc.Col(input_field("Annual Income ($)", dbc.Input(id='person_income', type='number', value=65000, className='form-control-modern')), width=4),
                            dbc.Col(input_field("Default on File (0/1)", dbc.Input(id='cb_person_default_on_file', type='number', value=0, min=0, max=1, step=1, className='form-control-modern')), width=4),
                            dbc.Col(input_field("Loan % of Income", dbc.Input(id='loan_percent_income', type='number', value=0.23, step=0.01, className='form-control-modern')), width=4),
                        ])
                    ], className='section-card mb-4'),

                    html.Div([
                        html.Div([
                            html.Span('💰', style={'fontSize': '1.2rem', 'marginRight': '0.5rem'}),
                            html.Span('Loan Details', className='section-title')
                        ], className='section-header mb-3'),

                        dbc.Row([
                            dbc.Col(input_field("Loan Amount ($)", dbc.Input(id='loan_amnt', type='number', value=15000, className='form-control-modern')), width=4),
                            dbc.Col(input_field("Interest Rate (%)", dbc.Input(id='loan_int_rate', type='number', value=10.5, step=0.1, className='form-control-modern')), width=4),
                            dbc.Col(input_field("Loan Intent", 
                                dcc.Dropdown(
                                    options=[
                                        {'label': 'Personal', 'value': 'PERSONAL'},
                                        {'label': 'Education', 'value': 'EDUCATION'},
                                        {'label': 'Home Improvement', 'value': 'HOMEIMPROVEMENT'}, # Note the spelling change
                                        {'label': 'Medical', 'value': 'MEDICAL'},
                                        {'label': 'Venture', 'value': 'VENTURE'},
                                        {'label': 'Debt Consolidation', 'value': 'DEBTCONSOLIDATION'}
                                    ],
                                    value='PERSONAL',
                                    id='loan_intent',
                                    className='dropdown-modern',
                                    clearable=False
                                )
                            ), width=4),
                        ]),
                        dbc.Row([
                            dbc.Col(input_field("Home Ownership", 
                                dcc.Dropdown(
                                    options=[
                                        {'label': 'Rent', 'value': 'RENT'},
                                        {'label': 'Mortgage', 'value': 'MORTGAGE'},
                                        {'label': 'Own', 'value': 'OWN'}
                                    ],
                                    value='MORTGAGE',
                                    id='person_home_ownership',
                                    className='dropdown-modern',
                                    clearable=False
                                )
                            ), width=12),
                        ], className='mt-3')
                    ], className='section-card mb-4'),

                    html.Div([
                        dbc.Button("RUN RISK ASSESSMENT", id='predict-btn', className='submit-btn-modern w-100')
                    ], className="mt-4")

                ], className='applicant-profile-card p-4')
            ], lg=6, md=12)

        ])  
    ], className='page-wrapper'),  

    dbc.Container(id='results-output', fluid=True)
])  

def get_shap_image(filename):
   
    return f"/assets/figures/{filename}"
    

@app.callback(
    Output('results-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    [State('person_age', 'value'), 
     State('person_income', 'value'), 
     State('loan_amnt', 'value'),
     State('person_home_ownership', 'value'), 
     State('loan_int_rate', 'value'),
     State('loan_intent', 'value'), 
     State('person_emp_length', 'value'),
     State('cb_person_cred_hist_length', 'value'), 
     State('cb_person_default_on_file', 'value'),
     State('loan_percent_income', 'value')],
    prevent_initial_call=True
)
def run_assessment(n_clicks, *args):
  
    (person_age, person_income, loan_amnt, person_home_ownership, 
     loan_int_rate, loan_intent, person_emp_length, 
     cb_person_cred_hist_length, cb_person_default_on_file, 
     loan_percent_income) = args

    preprocessor_columns = [
        'person_age', 'person_income', 'person_home_ownership', 'person_emp_length',
        'loan_intent', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
        'cb_person_default_on_file', 'cb_person_cred_hist_length'
    ]

    # 2.Map inputs 
    applicant = {
        'person_age': int(person_age) if person_age else 35,
        'person_income': float(person_income) if person_income else 65000,
        'person_home_ownership': str(person_home_ownership),
        'person_emp_length': float(person_emp_length) if person_emp_length else 3,
        'loan_intent': str(loan_intent),
        'loan_amnt': float(loan_amnt) if loan_amnt else 15000,
        'loan_int_rate': float(loan_int_rate) if loan_int_rate else 10.5,
        'loan_percent_income': float(loan_percent_income) if loan_percent_income else 0.23,
        # Model expects 'Y'/'N' based on your pkl categories 
        'cb_person_default_on_file': 'Y' if cb_person_default_on_file == 1 else 'N',
        'cb_person_cred_hist_length': float(cb_person_cred_hist_length) if cb_person_cred_hist_length else 5,
    }
    
    try:
        #Creating DataFrame 
        X_input = pd.DataFrame([applicant])[preprocessor_columns]
        X_processed = preprocessor.transform(X_input)
        
        #Model Prediction 
        pred_proba = model.predict_proba(X_processed)[0]
        pred_idx = model.predict(X_processed)[0]
        pred_label = 'Default Risk' if pred_idx == 1 else 'No Default'
        
        #Formatting results (same as your visual logic)
        risk_label, risk_color = RISK_LEVELS.get(pred_label, ('Unknown', '#2563eb'))
        #Transform and Predict
        X_processed = preprocessor.transform(X_input)
        
        pred_proba = model.predict_proba(X_processed)[0]
        pred_idx = model.predict(X_processed)[0]
        
        X_numeric_only = X_processed[:, :7] 
        
        X_for_clustering = X_numeric_only[:, :6] 
        
        cluster_features = cluster_scaler.transform(X_for_clustering)
        cluster_id = kmeans.predict(cluster_features)[0]
       
        cluster_name = CLUSTER_LABELS.get(cluster_id, 'Unknown')
        
        #Gauge/Bar for Probabilities
        proba_fig = go.Figure(go.Bar(
            x=['No Default', 'Default Risk'],
            y=[pred_proba[0], pred_proba[1]],
            marker_color=[STAGE_COLORS.get('No Default', '#58427C'), STAGE_COLORS.get('Default Risk', '#800080')],
            text=[f'{p*100:.1f}%' for p in pred_proba],
            textposition='auto',
        ))
        proba_fig.update_layout(title='Prediction Probabilities', template='plotly_white', height=300, margin=dict(t=40, b=20, l=40, r=20))
        
        #Feature Importance Graph
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        top_features = [preprocessor_columns[i] for i in indices if i < len(preprocessor_columns)]
        top_vals = [importances[i] for i in indices if i < len(preprocessor_columns)]
        
        importance_fig = go.Figure(go.Bar(x=top_vals[::-1], y=top_features[::-1], orientation='h', marker_color='#2563eb'))
        importance_fig.update_layout(title='Top Feature Drivers', template='plotly_white', height=300, margin=dict(t=40, b=20, l=100, r=20))

        #Recommendations Logic
        if pred_label == 'Default Risk':
            rec_text = "CAUTION: High risk of default detected. Recommend further manual review."
        else:
            rec_text = f"APPROVAL RECOMMENDED: Stable profile within the '{cluster_name}' segment."

        #Helper for SHAP images
        def create_shap_img(filename, title):
            img_path = get_shap_image(filename)
            return dbc.Col([
                html.P(title, className='desc-heading', style={'fontWeight': 'bold', 'marginTop': '10px'}),
                html.Img(src=img_path, style={'width': '100%', 'borderRadius': '8px', 'border': '1px solid #ddd'})
            ], md=6)

        #Assemble the Dashboard
        return html.Div([
            html.Div('Detailed Assessment Report', className='card-title', style={'fontSize': '24px', 'marginBottom': '20px'}),
            
            #Top 
            html.Div([
                html.Div([html.P('Prediction'), html.H3(pred_label, style={'color': risk_color})], className='stat-card', style={'flex': '1', 'padding': '20px', 'borderLeft': f'5px solid {risk_color}', 'background': '#f9f9f9', 'margin': '10px'}),
                html.Div([html.P('Confidence'), html.H3(f"{pred_proba[pred_idx]*100:.1f}%")], className='stat-card', style={'flex': '1', 'padding': '20px', 'borderLeft': '5px solid #2563eb', 'background': '#f9f9f9', 'margin': '10px'}),
                html.Div([html.P('Segment'), html.H3(cluster_name)], className='stat-card', style={'flex': '1', 'padding': '20px', 'borderLeft': '5px solid #58427C', 'background': '#f9f9f9', 'margin': '10px'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),

            #Middle
            html.Div([
                html.P("Recommendation:", style={'fontWeight': 'bold'}),
                html.P(rec_text)
            ], style={'background': '#eef2ff', 'padding': '20px', 'borderRadius': '10px', 'margin': '20px 10px'}),

            #Graphs
            dbc.Row([
                dbc.Col(dcc.Graph(figure=proba_fig), md=6),
                dbc.Col(dcc.Graph(figure=importance_fig), md=6)
            ]),

            #SHAP Analysis 
            html.Div([
                html.H4("Visual Explanations (SHAP Values)", style={'marginTop': '40px', 'borderTop': '1px solid #eee', 'paddingTop': '20px'}),
                dbc.Row([
                    create_shap_img('shap_bar.png', 'Global Importance'),
                    create_shap_img('shap_waterfall.png', 'Individual Driver Analysis')
                ]),
                dbc.Row([
                    create_shap_img('shap_beeswarm.png', 'Feature Impact Distribution'),
                    create_shap_img('cluster_pca.png', 'Clustering Visualization')
                ], className='mt-4')
            ])
        ], className='results-card', style={'padding': '30px', 'background': 'white', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'borderRadius': '15px'})

    except Exception as e:
            import traceback
            return html.Div([
                html.Div('Critical Assessment Error', className='card-title'),
                html.Pre(traceback.format_exc())
            ])


if __name__ == '__main__':
    app.run(debug=True)