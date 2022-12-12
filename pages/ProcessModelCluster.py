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
import dash_interactive_graphviz

# store for number of discovered models
models_num = dcc.Store(id='discovered-models-num', storage_type='local')
# conformance measure dropdown
conformance_dropdown = dcc.Dropdown(id='conformance-measure-cluster', options=['Fitness', 'Precision'], multi=False, value='Fitness')
# cluster dropdown
cluster_dropdown = dcc.Dropdown(id='cluster-selection', options=[], multi=False, disabled=True)

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
        html.Br(),
        dbc.Row([
            dbc.Col([html.Div("Select Cluster: ")], align='center'),
            dbc.Col([cluster_dropdown], align='center')
            ]),
        html.Div(
            [
                dash_interactive_graphviz.DashInteractiveGraphviz(id="ocpn-clustered-ocel")
            ]
            ),
    ]),
])

@app.callback(Output("cluster-selection", "options"), [State("clustered-ocels", "data")], [Input("cluster-selection", "disabled")], prevent_initial_call=True)
def on_clustering(clustered_ocel, btn_status):
    if btn_status == False:
        if clustered_ocel is None:
            raise PreventUpdate
        else:
            num_clusters = len(clustered_ocel)
            cluster_options=[{'label':i, 'value':i} for i in range(0, num_clusters)]
            return cluster_options
    else:
        raise PreventUpdate

@app.callback(Output("cluster-selection", "disabled"), [Input("start-pm_cluster", "n_clicks")], prevent_initial_call=True)
def on_button_click(n):
    if n > 0:
        return False 
    else:
        return True 

@app.callback(Output("ocpn-clustered-ocel", "dot_source"), [State("clustered-ocels", "data")], [Input("cluster-selection", "value")], prevent_initial_call=True)
def on_selection(clustered_ocel, selected_cluster):
    if selected_cluster != None:

        selected_cluster_ocel = clustered_ocel[int(selected_cluster)]
        clustered_ocel_obj = pickle.loads(codecs.decode(selected_cluster_ocel.encode(), "base64"))
        # discover petri nets
        ocpn = process_discovery.process_discovery_ocel_to_ocpn(clustered_ocel_obj)
        # graph source
        graphviz_src = process_discovery.ocpn_to_gviz(ocpn)

        return graphviz_src
    else:
        return dash.no_update


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