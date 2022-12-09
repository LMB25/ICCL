# Import necessary libraries 
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from app import app
from functions import process_discovery, conformance_checking
import base64
import dash
import pickle, codecs

# store for number of discovered models
models_num = dcc.Store(id='discovered-models-num', storage_type='local')
# conformance measure dropdown
conformance_dropdown = dcc.Dropdown(id='conformance-measure-cluster', options=['Fitness', 'Precision'], multi=False, value='Fitness')


# Define the page layout
layout = dbc.Container([
    dbc.Col([
        html.Center(html.H1("Process Discovery Clustered OCEL")),
        models_num,
        html.Hr(),
        dbc.Row([
            dbc.Col([html.Div("Select Conformance Measure:")], align='center'),
            dbc.Col([conformance_dropdown], align='center'),
            dbc.Col([dbc.Button("Calculate", id="calc-conformance-cluster", className="me-2", n_clicks=0)], align='center'),
            dbc.Col([html.Div("Result:")], align='center'),
            dbc.Col([html.Div(id='conformance-result-cluster')], align='center'),
        ]),
        html.Hr(),
        dbc.Row([
            dbc.Button("Show Process Models", color="warning", className="me-1", id='start-pm_cluster', n_clicks=0),
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


# display discovered sub-petri-nets
@app.callback(Output("pm-models-display", "children"), [State("discovered-models-num", "data")], Input("pm-models", "children"), prevent_initial_call=True)
def on_discovery(num_models, success_div):
    if success_div is None:
        raise PreventUpdate 
    else:
        imgs = []
        for i in range(0,num_models):
            test_base64 = base64.b64encode(open('imgs/oc_petri_net_cluster_' + str(i) + '.png', 'rb').read()).decode('ascii')
            imgs.append(html.H4("Cluster " + str(i)))
            imgs.append(html.Img(src='data:image/png;base64,{}'.format(test_base64), style={'height':'70%', 'width':'70%'}))
        return imgs

@app.callback(Output("conformance-result-cluster", "children"), [State("clustered-ocels", "data"), State("conformance-measure-cluster", "value")], [Input("calc-conformance-cluster", "n_clicks")])
def on_button_click(clustered_ocels, conformance_meas, n):
    if n > 0:
        conformance_results = []
        for ocel_log in clustered_ocels:
            # load ocel
            ocel_log = pickle.loads(codecs.decode(ocel_log.encode(), "base64"))
            # discover petri net
            ocpn = process_discovery.process_discovery_ocel_to_ocpn(ocel_log)
            # calculate fitness or precision, return as html div
            if conformance_meas == 'Fitness':
                fitness = conformance_checking.calculate_fitness(ocel_log, ocpn)
                conformance_results.append(fitness)
            elif conformance_meas == 'Precision':
                precision = conformance_checking.calculate_precision(ocel_log, ocpn)
                conformance_results.append(precision)
        # create conformance results Dataframe
        conformance_df = conformance_checking.create_conformance_df(conformance_results, conformance_meas)
        return dbc.Table.from_dataframe(conformance_df, striped=True, bordered=True, hover=True, id="conformance-cluster-table")
    else:
        return dash.no_update