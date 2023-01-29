# Import necessary libraries 
from dash import html, dcc
from dash.dependencies import Input, Output

# Connect to main app.py file
from app import app

# Connect to app pages
from pages import Dataimport, Dataview, Configuration, ProcessModel, ProcessModelCluster, Manual

# Connect the navbar/ sidebar to the index
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

# Define store objects to keep content while switching pages
# OCEL 
ocel_obj = dcc.Store(id='ocel_obj', storage_type='memory')
# OCEL parameters
ocel_params = dcc.Store(id='param-store', storage_type='local')
# Process Executions
ocel_executions = dcc.Store(id='execution-store', storage_type='memory')
# OCEL of Clusters
clustered_ocel_store = dcc.Store(id='clustered-ocels', storage_type='memory')
# List of lists of average Process Execution Features for each cluster
extracted_pe_features_cluster = dcc.Store(id='extracted-pe-features-cluster-store', storage_type='memory')

# Define the index page layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    nav, side, ocel_obj, ocel_params, ocel_executions, clustered_ocel_store, extracted_pe_features_cluster,
    html.Div(id='page-content', children=[], style=CONTENT_STYLE), 
])

# Page navigation
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def render_page_content(pathname):
    if pathname in ["/", "/page-1/1"]:
        return Dataimport.layout
    elif pathname == "/page-1/2":
        return Dataview.layout
    elif pathname == "/page-2/1":
        return Configuration.layout
    elif pathname == "/page-3/1":
        return ProcessModel.layout
    elif pathname == "/page-3/2":
        return ProcessModelCluster.layout
    elif pathname == "/page-4/1":
        return Manual.layout
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
    #host and port must be specified in order for docker to work
    app.run_server(debug=True, host='0.0.0.0', port=8050, use_reloader=False)
    #app.run_server(debug=True, use_reloader = True)



