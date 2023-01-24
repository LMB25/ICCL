# Import necessary libraries 
from dash import html, dcc
import dash_bootstrap_components as dbc
from app import app

# embed html file of user manual in Iframe                          
user_manual = html.Iframe(id='embedded_manual', src=app.get_asset_url('user_manual.html'), style={'width':'100%', 'height':'100vh'})

# Define the page layout
layout = dbc.Container([
    dbc.Row([
        html.Center(html.H1("User Manual")),
        html.Br(),
        html.Hr(),
        html.Br()
    ]),
    html.Div([user_manual])
])