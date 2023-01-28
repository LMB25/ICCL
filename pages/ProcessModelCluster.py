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
import os
import dash_interactive_graphviz
import pandas as pd

# store for number of discovered models
models_num = dcc.Store(id='discovered-models-num', storage_type='local')
# conformance measure dropdown
conformance_dropdown = dcc.Dropdown(id='conformance-measure-cluster', options=['Fitness', 'Precision'], multi=False, value='Fitness')
# cluster dropdown
cluster_dropdown = dcc.Dropdown(id='cluster-selection', options=[], multi=False, disabled=True)
# empty DataTable for Process Execution Features
feature_options_extraction_renamed = ["Number of Events", "Number of Ending Events", "Throughput Duration", "Number of Objects", "Unique Activities", "Number of Starting Events", "Duration of Last Event"]
dummy_df = pd.DataFrame(columns=feature_options_extraction_renamed)
feature_table = dbc.Table.from_dataframe(dummy_df, striped=True, bordered=True, hover=True)

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
            dbc.Button("Start Process Discovery", color="warning", className="me-1", id='start-pm_cluster', n_clicks=0),
        ]),
        html.Div(id='discovery-success-cluster'),
        html.Hr(),
        dbc.Row([
            dbc.Col([html.Div("Select Cluster: ")], align='center'),
            dbc.Col([cluster_dropdown], align='center')
            ]),
        html.Div([
            html.Hr(),
            dbc.Row([html.H5("Process Execution Features Average:")]),
            dbc.Row([html.Div(feature_table, id="pe-feature-avg-table")]), 
            ], id='avg-pe-features', style={'display': 'block'}),
        html.Div([
            html.Hr(),
            dbc.Row([
                dbc.Col([html.H5("Save Petri Net")], align='center'),
                dbc.Col([html.Div("Filename: ")], align='center'),
                dbc.Col([dbc.Input(id='filename-clustered-model', placeholder='cluster_model')], width=3),
                dbc.Col([dcc.Dropdown(id='img-format-cluster', options=['png', 'svg'], multi=False, value='png')], align='center'),
                dbc.Col([dbc.Button("Save", id="save-img-clustered-model", className="me-2", n_clicks=0)]),
                dcc.Download(id="download-image-cluster")
            ]), 
            ], id='download-model-clustered', style={'display': 'none'}),
        html.Div("Petri Net successfully saved.", id='download-success-cluster', style={'display':'none'}),
        html.Hr(),
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

@app.callback([Output("cluster-selection", "disabled"), Output("discovery-success-cluster", "children")], [Input("start-pm_cluster", "n_clicks")], prevent_initial_call=True)
def on_button_click(n):
    if n > 0:
        return False, ["Process Models successfully discovered."]
    else:
        return True 

@app.callback([Output("download-model-clustered", "style"), Output("ocpn-clustered-ocel", "dot_source"), Output("pe-feature-avg-table", "children")], [State("clustered-ocels", "data"), State("extracted-pe-features-cluster-store","data")], 
              [Input("cluster-selection", "value")], prevent_initial_call=True)
def on_selection(clustered_ocel, avg_pe_features_list, selected_cluster):
    if selected_cluster != None:

        selected_cluster_ocel = clustered_ocel[int(selected_cluster)]
        clustered_ocel_obj = pickle.loads(codecs.decode(selected_cluster_ocel.encode(), "base64"))
        # discover petri nets
        ocpn = process_discovery.process_discovery_ocel_to_ocpn(clustered_ocel_obj)
        # graph source
        graphviz_src = process_discovery.ocpn_to_gviz(ocpn)
        # avg process execution features
        if avg_pe_features_list != None:
            df_extr = pd.DataFrame(columns=["Feature", "Value"])
            avg_pe_features = avg_pe_features_list[int(selected_cluster)]
            df_extr["Feature"] = feature_options_extraction_renamed
            df_extr["Value"] = avg_pe_features
            df_transposed = df_extr.T
            df_transposed.columns = df_transposed.iloc[0]
            df_transposed = df_transposed[1:]
            datatable = dbc.Table.from_dataframe(df_transposed, striped=True, bordered=True, hover=True)
        else:
            datatable = feature_table

        return {'display':'block'}, graphviz_src, datatable
    else:
        return dash.no_update

# to-do: don't load and discover again -> store discovered ocpn beforehand
@app.callback([Output("download-success-cluster", "style"), Output("download-image-cluster", "data")], [State("cluster-selection", "value"), State("clustered-ocels", "data"), State("filename-clustered-model", "value"), State("img-format-cluster", "value")], Input("save-img-clustered-model", "n_clicks"), prevent_initial_call=True)
def on_download_click(selected_cluster, clustered_ocel, filename, format, n):
    if n > 0:
        selected_cluster_ocel = clustered_ocel[int(selected_cluster)]
        clustered_ocel_obj = pickle.loads(codecs.decode(selected_cluster_ocel.encode(), "base64"))
        # discover petri net
        ocpn = process_discovery.process_discovery_ocel_to_ocpn(clustered_ocel_obj)
        # save petri net
        process_discovery.save_ocpn(ocpn, 'assets/process_models/', filename, format)
        return {'display':'block'}, dcc.send_file('assets/process_models/' + filename + '.' + format)
    else:
        dash.no_update


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

