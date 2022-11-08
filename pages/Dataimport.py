# Import necessary libraries 
from dash import html
import dash_bootstrap_components as dbc

from components import upload

upload_field = upload.Upload_OCEL()

# Define the page layout
layout = dbc.Container([
    dbc.Row([
        html.Center(html.H1("Import Data")),
        html.Br(),
        html.Hr(),
        dbc.Col([
            html.P("List of uploaded OCEL"), 
            html.P("< Table >"),
        ]),
        dbc.Col([
            html.P("Import OCEL"), 
            upload_field
        ]), 
    ])
])