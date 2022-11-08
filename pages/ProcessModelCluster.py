# Import necessary libraries 
from dash import html
import dash_bootstrap_components as dbc


# Define the page layout
layout = dbc.Container([
    dbc.Row([
        html.Center(html.H1("Process Discovery Clustered OCEL")),
        html.Br(),
        html.Hr(),
        html.P("Process Models of clustered OCEL")
    ])
])