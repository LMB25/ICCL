# Import necessary libraries 
from dash import html, dcc
import dash_bootstrap_components as dbc
from app import app
import base64

markdown_test = dcc.Markdown('''
                                # This is the user manual
                                Let's add some information
                                * information 1
                                * information 2

                                ## Section 1
                                Let's try to add an image
                            ''')
                            
# Define the page layout
layout = dbc.Container([
    dbc.Row([
        html.Center(html.H1("User Manual")),
        html.Br(),
        html.Hr(),
        html.Br()
    ]),
    dbc.Row([
        html.Div([markdown_test])
    ]),
    dbc.Row([
        html.Img(src=app.get_asset_url('test.png'))
    ]),
])