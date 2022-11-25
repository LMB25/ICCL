# Import necessary libraries 
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from app import app
from functions import process_discovery
import base64
import dash
import pickle, codecs

# number of clicks store
models_store = dcc.Store(id='discovered-models', storage_type='local')
# store for number of discovered models
models_num = dcc.Store(id='discovered-models-num', storage_type='local')

# Define the page layout
layout = dbc.Container([
    dbc.Col([
        html.Center(html.H1("Process Discovery Clustered OCEL")),
        html.Br(),
        models_store, models_num,
        html.Hr(),
        dbc.Row([
            dbc.Button("Start Process Discovery", color="warning", className="me-1", id='start-pm_cluster', n_clicks=0),
        ]),
        html.Div(id='pm-models'),
        html.Div(id='pm-models-display')
    ])
])

@app.callback([Output("pm-models", "children"), Output("discovered-models-num", "data")], State("clustered-ocels", "data"), [Input("start-pm_cluster", "n_clicks")])
def on_button_click(clustered_ocel, n):
    if n > 0:
        # load ocel
        for i, sub_ocel in enumerate(clustered_ocel):
            clustered_ocel_obj = pickle.loads(codecs.decode(sub_ocel.encode(), "base64"))
            # discover and save petri nets
            process_discovery.process_discovery_ocel_to_img(clustered_ocel_obj, "oc_petri_net_cluster_" + str(i))
        return "Process Models successfully discovered.", i+1
    else:
        return dash.no_update


# count number of clicks
@app.callback(Output("discovered-models", "data"), State("discovered-models", "data"), Input("start-pm_cluster", "n_clicks"))
def on_click(data, n_clicks):
    if n_clicks == 0:
        raise PreventUpdate
    data = data or {'clicks': 0}
    data['clicks'] = data['clicks'] + 1
    return data

# display discovered sub-petri-nets
@app.callback(Output("pm-models-display", "children"), Input("discovered-models-num", "data"), prevent_initial_call=True)
def on_discovery(num_models):
    if num_models != None:
        imgs = []
        for i in range(0,num_models):
            test_base64 = base64.b64encode(open('imgs/oc_petri_net_cluster_' + str(i) + '.png', 'rb').read()).decode('ascii')
            imgs.append(html.H4("Cluster " + str(i+1)))
            imgs.append(html.Img(src='data:image/png;base64,{}'.format(test_base64), style={'height':'70%', 'width':'70%'}))
        return imgs