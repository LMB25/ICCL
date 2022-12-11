# Import necessary libraries 
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash
import time
from app import app
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Dash, Trigger, ServersideOutput
from functions import process_executions, feature_extraction, graph_embedding, clustering, dataimport
from components import nxgraph_figure
import pickle
import codecs

# execution feature set store
execution_store = dcc.Store('execution-feature-set', storage_type='local')
# event feature set store
event_store = dcc.Store('event-feature-set', storage_type='local')


# options for event based feature selection
feature_options_event = ['EVENT_REMAINING_TIME', 'EVENT_ELAPSED_TIME', 'EVENT_FLOW_TIME']
# options for extraction based feature selection
feature_options_extraction = ['EXECUTION_NUM_OF_EVENTS', 'EXECUTION_NUM_OF_END_EVENTS', 'EXECUTION_THROUGHPUT', 'EXECUTION_NUM_OBJECT', 'EXECUTION_UNIQUE_ACTIVITIES', 
                              'EXECUTION_NUM_OF_STARTING_EVENTS', 'EXECUTION_LAST_EVENT_TIME_BEFORE']
                        
# event based feature selection dropdown
event_feature_selection_dropdown= dcc.Dropdown(id='feature-selection-event', options=[{'label': i, 'value': i} for i in feature_options_event], multi=True, value=feature_options_event)
# extraction based feature selection dropdown
extraction_feature_selection_dropdown= dcc.Dropdown(id='feature-selection-extraction', options=[{'label': i, 'value': i} for i in feature_options_extraction], multi=True, value=feature_options_extraction)

# Define the page layout
layout = dbc.Container([
        execution_store, event_store, 
        html.Center(html.H1("Clustering")),
        html.Hr(),
        dbc.Row([
            dbc.Col([html.Div("Number of Process Executions:"), html.Div(id="process-executions-summary")]),
            dbc.Col([html.Div("Select Process Execution to show Graph"), html.Div(dcc.Dropdown(id='executions_dropdown'))])
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col([html.Div("Select Event Features for Clustering:"), html.Div(event_feature_selection_dropdown)]),
            dbc.Col([html.Div("Select Extraction Features for Clustering:"), html.Div(extraction_feature_selection_dropdown)])
        ]),
        html.Br(),
        dbc.Row([dbc.Button("Set Selected Features", className="me-2", id='set-features', n_clicks=0)]),
        html.Div(id='feature-sucess'),
        html.Br(),
        dbc.Row([
            dbc.Col([html.Div("Select Graph Embedding Method"), html.Div(dcc.Dropdown(['Graph2Vec', 'Feather-G'], 'Feather-G',id='graph-embedding-dropdown'))]),
            dbc.Col([html.Div("Select Clustering Technique"), html.Div(dcc.Dropdown(['K-Means', 'Mean-Shift', 'Hierarchical'],'Mean-Shift',id='clustering-method-dropdown'))]),
            dbc.Col([html.Div("Select Number of Clusters"), html.Div(dcc.Slider(1,7,1, value=2, id='num-clusters-slider', disabled=True))])
        ]),
        html.Br(),
        dbc.Row([dbc.Button("Start Clustering", color="warning", className="me-1", id='start-clustering', n_clicks=0)]),
        html.Div(id='clustering-success'),
        html.Br(),
        html.Div(id="cluster-summary-component"),
        html.Br(),
        html.Div(id='process-execution-graph'),
])

# load selected features in stores
@app.callback([Output("event-feature-set", "data"), Output("execution-feature-set", "data"), Output("feature-sucess", "children")], [State("feature-selection-event", "value"), State("feature-selection-extraction", "value")], Input("set-features", "n_clicks"))
def on_click(selected_event_features, selected_execution_features, n_clicks):
    if n_clicks > 0:
        # set selected event features
        feature_set_event = selected_event_features
        # set selected execution features
        feature_set_extraction = selected_execution_features
        return feature_set_event, feature_set_extraction, "Features successfully set."
    else:
        raise PreventUpdate

# update disability status of number of clusters slider
@app.callback(Output('num-clusters-slider', 'disabled'), Input('clustering-method-dropdown', 'value'))
def on_change_clustering_method(clustering_method):
    if clustering_method in ['K-Means', 'Hierarchical']:
        return False 
    else:
        return True

# perform clustering and return dataframe with process execution ids and cluster labels
@app.callback([ServersideOutput("clustered-ocels", "data"), Output("clustering-success", "children"), Output("cluster-summary-component", "children")], 
                [State("ocel_obj", "data"), State("event-feature-set", "data"), State("execution-feature-set", "data"), State("graph-embedding-dropdown", "value"), State('clustering-method-dropdown', 'value'), State('num-clusters-slider', 'value')], 
                Trigger("start-clustering", "n_clicks"), memoize=True)
def on_click(ocel_log, selected_event_features, selected_execution_features, embedding_method, clustering_method, num_clusters, n_clicks):
    time.sleep(1)
    if n_clicks > 0:
        # load ocel
        ocel_log = pickle.loads(codecs.decode(ocel_log.encode(), "base64"))
        # extract features, get feature graphs
        feature_storage = feature_extraction.extract_features(ocel_log, selected_event_features, selected_execution_features, 'graph')
        # remap nodes of feature graphs
        feature_nx_graphs = graph_embedding.feature_graphs_to_nx_graphs(feature_storage.feature_graphs)
        # embedd feature graphs
        if embedding_method == 'Graph2Vec':
            embedding = graph_embedding.perform_graph2vec(feature_nx_graphs, False)
        elif embedding_method == 'Feather-G':
            embedding = graph_embedding.perform_feather_g(feature_nx_graphs)
        # cluster embedding
        if clustering_method == 'Mean-Shift':
            labels = clustering.perform_MeanShift(embedding)
        elif clustering_method == 'K-Means':
            labels = clustering.perform_KMeans(embedding, num_clusters)
        elif clustering_method == 'Hierarchical':
            labels = clustering.perform_HierarchicalClustering(embedding, num_clusters)
        # create Dataframe with process execution id and cluster labels
        clustered_df = clustering.create_clustered_df(ocel_log.process_executions, labels)
        # get summary of clusters
        cluster_summary_df = clustering.get_cluster_summary(clustered_df)
        # partition ocel into clustered ocels
        ocel_df, _ = dataimport.ocel_to_df_params(ocel_log)
        sub_ocels = clustering.partition_ocel(ocel_log, ocel_df, clustered_df)
        # encoding/ storing of sub ocels
        sub_ocels_encoded = [codecs.encode(pickle.dumps(ocel), "base64").decode() for ocel in sub_ocels]
    else:
        raise PreventUpdate
    return sub_ocels_encoded, "Clustering successfully performed.", dbc.Table.from_dataframe(cluster_summary_df, striped=True, bordered=True, hover=True, id="cluster-summary-table")

@app.callback([Output("process-executions-summary", "children"), Output("executions_dropdown", "options")], [Input("execution-store", "data")])
def on_extraction(executions):
    if executions != None:
        # update display of number of process executions
        executions_summary_text = html.P([str(len(executions))])
        # update dropdown widget with list of process executions
        options = [{"label":str(i+1),"value":str(i+1)} for i in range(0,len(executions))]
        return executions_summary_text, options


@app.callback(Output("process-execution-graph", "children"), State("ocel_obj", "data"), [Input("executions_dropdown", "value")])
def on_extraction(ocel_log, execution_id):
    if execution_id != None:
        # load ocel
        ocel_log = pickle.loads(codecs.decode(ocel_log.encode(), "base64"))
        # get graph object of process execution
        ocel_executions_graph = process_executions.get_process_execution_graph(ocel_log, int(execution_id) - 1)
        # convert nx graph to dash figure 
        fig = nxgraph_figure.create_graph_figure(ocel_executions_graph)

        return dcc.Graph(id='pe-graph',figure=fig)
