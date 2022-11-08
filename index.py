# Import necessary libraries 
from dash import html, dcc
from dash.dependencies import Input, Output, State
import base64, io
from functions import dataimport

# Connect to main app.py file
from app import app

# Connect to your app pages
from pages import Dataimport, Dataview, Configuration, ProcessModel, ProcessModelCluster

# Connect the navbar to the index
from components import sidebar, navbar

# Define the navbar
nav = navbar.Navbar()

# Define the sidebar
side = sidebar.Sidebar()

# Content style
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

# Define the index page layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    nav, side,
    html.Div(id='page-content', children=[], style=CONTENT_STYLE), 
])

# Page navigation
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def render_page_content(pathname):
    if pathname in ["/", "/page-1/1"]:
        return Dataview.layout
    elif pathname == "/page-1/2":
        return Dataimport.layout
    elif pathname == "/page-2/1":
        return Configuration.layout
    elif pathname == "/page-3/1":
        return ProcessModel.layout
    elif pathname == "/page-3/2":
        return ProcessModelCluster.layout
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


# Run the app on localhost:8050
if __name__ == '__main__':
    app.run_server(debug=True)
