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
from components import nxgraph_figure, silhouette_figure
import pickle
import codecs
import pandas as pd


# event feature set store
event_store = dcc.Store('event-feature-set', storage_type='local')
# graph embedding parameters store
embedding_params_store = dcc.Store('embedding-parameters', storage_type='local')
# Define Store object for List of DataFrames of Process Execution Features
extracted_pe_features = dcc.Store(id='extracted-pe-features-store')
# clustering parameters store
clustering_params_store = dcc.Store('clustering-parameters', storage_type='local')

# options for event based feature selection
feature_options_event = ['EVENT_REMAINING_TIME', 'EVENT_ELAPSED_TIME', 'EVENT_FLOW_TIME', 'EVENT_ACTIVITY', 'EVENT_NUM_OF_OBJECTS', 'EVENT_PREVIOUS_ACTIVITY_COUNT', 'EVENT_DURATION']
                        
# event based feature selection dropdown
event_feature_selection_dropdown= dcc.Dropdown(id='feature-selection-event', options=[{'label': i, 'value': i} for i in feature_options_event], multi=True, value=[])#feature_options_event)

# empty DataTable for Process Execution Features
feature_options_extraction_renamed = ["Number of Events", "Number of Ending Events", "Throughput Duration", "Number of Objects", "Unique Activities", "Number of Starting Events", "Duration of Last Event"]
dummy_df = pd.DataFrame(columns=feature_options_extraction_renamed)
feature_table = dbc.Table.from_dataframe(dummy_df, striped=True, bordered=True, hover=True)

# silhouette analysis explanation
silhouette_explanation = dbc.Card(
                                dbc.CardBody("Intuitively, the silhouette score quantifies the space between different clusters. For each number of clusters (up to the inserted max.), the selected clustering method is performed. Afterwards, it is measured how similar the observation are to the assigned cluster and how dissimilar they are to the observation of the nearest cluster. The plot displays the average silhouette score for each number of clusters. The measure has the range [-1,+1], whereas a score near +1 indicates that the clusters are well separated and negative scores indicate that the samples might be wrongly separated. Thus, to get a reasonable clustering result, one should choose the cluster number with the maximal positive average silhouette score."),
                                className="mb-3",
                                )



# process executions explanation
process_executions_explanation = dbc.Card(
                                dbc.CardBody("Hier sollte eine Erklärung zu den Process Executions stehen. Was zeigt der Plot des Process Execution Graphs an?"),
                                className="mb-3",
                                )
# list of features and explanation
features_explanation = dbc.Card(
                                dbc.ListGroup(
                                    [
                                        dbc.ListGroupItem("EVENT_REMAINING_TIME: Remaining time from event to end of process execution."),
                                        dbc.ListGroupItem("EVENT_ELAPSED_TIME: Elapsed time from process execution start to the event."),
                                        dbc.ListGroupItem("EVENT_FLOW_TIME:"),
                                        dbc.ListGroupItem("EVENT_ACTIVITY: Activity that is performed in the event."),
                                        dbc.ListGroupItem("EVENT_NUM_OF_OBJECTS: Number of objects involved in the event."),
                                        dbc.ListGroupItem("EVENT_PREVIOUS_ACTIVITY_COUNT: Number of activities that took place before the event."),
                                        dbc.ListGroupItem("EVENT_DURATION: Duration of the event."),
                                    ],
                                    flush=True,
                                ),
                            )


# create form for attributed graph embedding parameters
embedding_params_form_attributed = html.Div([
        html.H5("Modify Graph Embedding Parameters:"),
        dbc.Row([
            dbc.Col(html.P("SVD Reduction Dimensions: ")),
            dbc.Col(dbc.Input(id='svd-dimensions', value=64))
        ]),
        dbc.Row([
            dbc.Col(html.P("SVD Iterations: ")),
            dbc.Col(dbc.Input(id='svd-iterations', value=20))
        ]),
        dbc.Row([
            dbc.Col(html.P("Maximal evaluation point: ")),
            dbc.Col(dbc.Input(id='theta-max', value=2.5))
        ]),
        dbc.Row([
            dbc.Col(html.P("Number of characteristic function evaluation points: ")),
            dbc.Col(dbc.Input(id='eval-points', value=25))
        ]),
        dbc.Row([
            dbc.Col(html.P("Scale - number of adjacency matrix powers: ")),
            dbc.Col(dbc.Input(id='order', value=5))
        ]),
        html.Br(),
        ], id='embedding-params-div-attributedgraph2vec', style={'display': 'none'})

# create form for Graph2Vec embedding parameters
embedding_params_form_graph2vec = html.Div([
        html.H5("Modify Graph Embedding Parameters:"),
        dbc.Row([
            dbc.Col(html.P("WL iterations: ")),
            dbc.Col(dbc.Input(id='wl-iterations', value=50))
        ]),
        dbc.Row([
            dbc.Col(html.P("Dimensions: ")),
            dbc.Col(dbc.Input(id='graph2vec-dim', value=128))
        ]),
        dbc.Row([
            dbc.Col(html.P("Epochs: ")),
            dbc.Col(dbc.Input(id='epochs', value=10))
        ]),
        dbc.Row([
            dbc.Col(html.P("Learning Rate: ")),
            dbc.Col(dbc.Input(id='learning-rate', value=0.025))
        ]),
        html.Br(),
        ], id='embedding-params-div-graph2vec', style={'display': 'none'})

# create form for Feather-G embedding parameters
embedding_params_form_featherg = html.Div([
        html.H5("Modify Graph Embedding Parameters:"),
        dbc.Row([
            dbc.Col(html.P("Maximal evaluation point: ")),
            dbc.Col(dbc.Input(id='theta-max-g', value=2.5))
        ]),
        dbc.Row([
            dbc.Col(html.P("Number of characteristic function evaluation points: ")),
            dbc.Col(dbc.Input(id='eval-points-g', value=25))
        ]),
        dbc.Row([
            dbc.Col(html.P("Scale - number of adjacency matrix powers: ")),
            dbc.Col(dbc.Input(id='order-g', value=5))
        ]),
        html.Br(),
        ], id='embedding-params-div-featherg', style={'display': 'none'})

# create form for K-Means parameters
clustering_params_form_kmeans = html.Div([
        dbc.Row([
            dbc.Col(html.P("Number of times the algorithm is run with different centroid seeds: ")),
            dbc.Col(dbc.Input(id='n-init-kmeans', value=10))
        ]),
        dbc.Row([
            dbc.Col(html.P("Maximum number of iterations for a single run: ")),
            dbc.Col(dbc.Input(id='max-iter-kmeans', value=300))
        ]),
        html.Br(),
        ], id='clustering-params-div-kmeans', style={'display': 'none'})

# create form for Hierarchical Clustering parameters
clustering_params_form_hierarchical = html.Div([
        dbc.Row([
            dbc.Col(html.P("Linkage criterion: ")),
            dbc.Col(dcc.Dropdown(options=['ward', 'complete', 'average', 'single'], id='linkage-criterion', value='ward'))
        ]),
        html.Br(),
        ], id='clustering-params-div-hierarchical', style={'display': 'none'})

# create form for Mean Shift parameters
clustering_params_form_meanshift = html.Div([
        dbc.Row([
            dbc.Col(html.P("Maximum number of iterations: ")),
            dbc.Col(dbc.Input(id='max-iter-meanshift', value=300))
        ]),
        html.Br(),
        ], id='clustering-params-div-meanshift', style={'display': 'none'})

# create form for Affinity-Propagation parameters
clustering_params_form_affinity = html.Div([
        dbc.Row([
            dbc.Col(html.P("Maximum number of iterations: ")),
            dbc.Col(dbc.Input(id='max-iter-affinity', value=200))
        ]),
        dbc.Row([
            dbc.Col(html.P("Number of iterations with no change in the number of estimated clusters that stops the convergence: ")),
            dbc.Col(dbc.Input(id='convergence-iter-affinity', value=15))
        ]),
        html.Br(),
        ], id='clustering-params-div-affinity', style={'display': 'none'})


# create empty div for embedding param form
embedding_param_form = html.Div([embedding_params_form_attributed, embedding_params_form_graph2vec, embedding_params_form_featherg], id='embedding-params-div', style={'display': 'block'})

# create empty div for clustering param form
clustering_param_form = html.Div([clustering_params_form_kmeans, clustering_params_form_hierarchical, clustering_params_form_meanshift, clustering_params_form_affinity], id='clustering-params-div', style={'display':'block'})

# Define the page layout
layout = dbc.Tabs([
        event_store, embedding_params_store, extracted_pe_features, clustering_params_store,
        dbc.Tab([
                html.Br(),
                html.H5("Feature Explanation:"),
                dbc.Row([dbc.Col([features_explanation], width=7),]),
                html.Hr(),
                dbc.Row(dbc.Col([html.H5("Feature Selection:"), html.Div("Select Event Features for Clustering:")], width=7)),
                dbc.Row([dbc.Col([html.Div(event_feature_selection_dropdown)], width=7), dbc.Col([dbc.Button("Set Selected Features", className="me-2", id='set-features', n_clicks=0), html.Div(id='feature-sucess')], width=5)])
        ], label="Features", tab_id='features-tab', label_style={'background-color': '#8daed9'}),
        dbc.Tab([
                html.Br(),
                dbc.Row([
                    dbc.Col([html.H5("Select Graph Embedding Method"), html.Div(dbc.RadioItems(options=[{"label": "AttributedGraph2Vec", "value": "AttributedGraph2Vec"},{"label": "Graph2Vec", "value": 'Graph2Vec'},{"label": "Feather-G", "value": "Feather-G"}], value="AttributedGraph2Vec", id="graph-embedding-selection"),)]),
                ]),
                html.Br(),
                dbc.Row([ 
                        dbc.Col([embedding_param_form], width=8),
                        dbc.Col([dbc.Button("Parse Embedding Parameters", id="parse-embedding-params", className="me-2", n_clicks=0)]),
                        html.Div("Parameters successfully parsed.", style={'display':'none'}, id='success-parse-embedding-params'),
                        ],
                        style={'display':'block'}, id='silhouette-div'),
                html.Br(),
                ], label="Embedding", tab_id='embedding-tab', label_style={'background-color': '#8dd996'}),
        dbc.Tab([
                html.Br(),
                dbc.Row([
                        dbc.Col([html.H5("Select Clustering Technique"), html.Div(dbc.RadioItems(options=[{"label": "K-Means", "value": "K-Means"},{"label": "Mean-Shift", "value": 'Mean-Shift'},{"label": "Hierarchical", "value": "Hierarchical"}, {"label": "Affinity-Propagation", "value":"AffinityPropagation"}], value="K-Means", id="clustering-method-selection"),)]),
                        ]),
                html.Br(),
                dbc.Row([dbc.Col([html.H5("Modify Clustering Embedding Parameters:")])]),
                dbc.Row([
                        dbc.Col([html.Div("Select Number of Clusters"), html.Div(dcc.Slider(1,10,1, value=2, id='num-clusters-slider', disabled=True)), clustering_param_form], width=7),
                        ]),
                html.Br(),
                dbc.Row([
                        dbc.Col([dbc.Button("Parse Clustering Parameters", id="parse-clustering-params", className="me-2", n_clicks=0)], width=4),
                        html.Div("Parameters successfully parsed.", style={'display':'none'}, id='success-parse-clustering-params'),
                        ]),
                html.Br(),
                dbc.Row([
                        dbc.Col([dbc.Button("Start Clustering", color="warning", className="me-1", id='start-clustering', n_clicks=0)], width=4)
                        ]),
                html.Div(id='clustering-success'),
                html.Br(),
                html.Div(id="cluster-summary-component"),
                html.Br(),
                ], label="Clustering", tab_id='clustering-tab', label_style={'background-color': '#f5b553'}),
        dbc.Tab([
                html.Br(),
                dbc.Row([
                    dbc.Col(process_executions_explanation)
                ]),
                dbc.Row([
                        dbc.Col([html.Div("Number of Process Executions:"), html.Div(id="process-executions-summary")]),
                        dbc.Col([dbc.Button("Start Process Execution Feature Extraction", color="warning", className="me-1", id='start-feature-extraction-pe', n_clicks=0), html.Div(id='pe-feature-success')]),
                        dbc.Col([html.Div("Select Process Execution:"), html.Div(dcc.Dropdown(id='executions_dropdown'))])
                    ]),
                html.Hr(),
                dbc.Row([html.Div(feature_table, id="pe-feature-table")]),
                html.Hr(),
                dbc.Row([html.Div(id='process-execution-graph')]),
                ], label="Process Executions", tab_id='process-executions-tab'),
        dbc.Tab([
                html.Br(),
                dbc.Row([
                    dbc.Col(silhouette_explanation)
                ]),
                html.Div([
                          dbc.Col([dbc.Alert([html.I(className="fa-solid fa-triangle-exclamation"),"You have to parse embedding parameters first."],color="warning", className="d-flex align-items-center")], width=4)
                         ], id='alert-params-silhouette', hidden=False),
                dbc.Row([
                    dbc.Col(html.Div("Select maximal number of clusters: "), width=3, align="center"),
                    dbc.Col(dbc.Input(id='max-clusters', placeholder='7'), width=1),
                    dbc.Col(dbc.Button("Apply Silhouette Analysis", className="me-2", id='start-silhouette', n_clicks=0, disabled=True), width=3),
                ]),
                dbc.Row([
                    dbc.Col(html.Div(id='silhouette-plot'))
                    ])
                ], label="Silhouette Analysis", tab_id='silhouette-tab'),
        ], id='configuration-tabs',active_tab='features-tab')

# switch between active tabs
@app.callback(Output('configuration-tabs', 'active_tab'), [Input("feature-sucess", "children"), Input("success-parse-embedding-params", "style")], prevent_initial_call=True)
def on_configuration(features_set, embedding_parsed):
    if features_set != None and embedding_parsed == {'display':'none'}:
        return 'embedding-tab'
    elif features_set != None and embedding_parsed == {'display':'block'}:
        return 'clustering-tab'
    else:
        dash.no_update

# load selected features in stores
@app.callback([Output("event-feature-set", "data"), Output("feature-sucess", "children")], State("feature-selection-event", "value"), Input("set-features", "n_clicks"))
def on_click(selected_event_features, n_clicks):
    if n_clicks > 0:
        # set selected event features
        feature_set_event = selected_event_features
        return feature_set_event, "Features successfully set."
    else:
        raise PreventUpdate

# extract process execution features, get DataFrames of features
@app.callback([ServersideOutput('extracted-pe-features-store', 'data'), Output("pe-feature-success", "children")], Trigger("start-feature-extraction-pe", "n_clicks"), State("ocel_obj", "data"), prevent_initial_call=True, memoize=True)
def on_click(n_clicks, ocel):
    if n_clicks > 0:
        # load ocel
        ocel = pickle.loads(codecs.decode(ocel.encode(), "base64"))
        list_feature_dfs = feature_extraction.create_extraction_feature_dfs(ocel)
        return list_feature_dfs, "Features successfully extracted."
    else:
        raise PreventUpdate

# update disability status of number of clusters slider, show option to apply silhouette analysis, if kmeans or hierarchical is selected
@app.callback([Output('num-clusters-slider', 'disabled'), Output('start-silhouette', 'disabled')], Input('clustering-method-selection', 'value'))
def on_change_clustering_method(clustering_method):
    if clustering_method in ['K-Means', 'Hierarchical']:
        disabled = False
        return False, disabled
    else:
        return True, True

# show alert if silhouette analysis is selected, but no embedding parameters are parsed
@app.callback(Output("alert-params-silhouette","hidden"), Input("embedding-parameters","data"), prevent_initial_call=True)
def on_embedding_parameters_parsed(embedding_data):
    if embedding_data != None:
        return True
    else:
        return False

# show parameter form for selected graph embedding method
@app.callback([Output("embedding-params-div-graph2vec", "style"), Output("embedding-params-div-attributedgraph2vec", "style"), Output("embedding-params-div-featherg", "style")], Input("graph-embedding-selection", "value"))
def on_embedding_selection(embedding_method):
    if embedding_method == 'Graph2Vec':
        return {'display':'block'}, {'display':'none'}, {'display':'none'}
    elif embedding_method == 'AttributedGraph2Vec':
        return {'display':'none'}, {'display':'block'}, {'display':'none'}
    elif embedding_method == "Feather-G":
        return {'display':'none'}, {'display':'none'}, {'display':'block'}
    else:
        return {'display':'none'}, {'display':'none'}, {'display':'none'}

# show parameter form for selected clustering method
@app.callback([Output("clustering-params-div-kmeans", "style"), Output("clustering-params-div-hierarchical", "style"), Output("clustering-params-div-meanshift", "style"), Output("clustering-params-div-affinity", "style")], Input("clustering-method-selection", "value"))
def on_clustering_selection(clustering_method):
    if clustering_method == 'K-Means':
        return {'display':'block'}, {'display':'none'}, {'display':'none'}, {'display':'none'}
    elif clustering_method == 'Hierarchical':
        return {'display':'none'}, {'display':'block'}, {'display':'none'}, {'display':'none'}
    elif clustering_method == "Mean-Shift":
        return {'display':'none'}, {'display':'none'}, {'display':'block'}, {'display':'none'}
    elif clustering_method == 'AffinityPropagation':
        return {'display':'none'}, {'display':'none'}, {'display':'none'}, {'display':'block'}
    else:
        return {'display':'none'}, {'display':'none'}, {'display':'none'}, {'display':'none'}

# save graph embedding parameter settings
@app.callback([Output("embedding-parameters", "data"), Output("success-parse-embedding-params", "style")], [State("svd-dimensions", "value"), State("svd-iterations", "value"), State("theta-max", "value"), State("eval-points", "value"), State("order", "value"),
                State("wl-iterations", "value"), State("graph2vec-dim", "value"), State("epochs", "value"), State("learning-rate", "value"), State("graph-embedding-selection", "value"), State("theta-max-g", "value"), State("eval-points-g", "value"), State("order-g", "value")], 
                Input("parse-embedding-params", "n_clicks"), prevent_initial_call=True)
def on_click_parse_params(svd_dimension, svd_iterations, theta_max, eval_points, order, wl_iterations, dimensions, epochs, learning_rate, embedding_method, theta_max_g, eval_points_g, order_g, n_click):
    if n_click > 0:
        if embedding_method =='AttributedGraph2Vec':
            embedding_params_dict = {"svd_dimensions":int(svd_dimension), "svd_iterations":int(svd_iterations), "theta_max":float(theta_max), "eval_points":int(eval_points), "order":int(order)}
            return embedding_params_dict, {'display':'block'}
        elif embedding_method == 'Graph2Vec':
            embedding_params_dict = {"wl_iterations":int(wl_iterations), "dimensions":int(dimensions), "epochs":int(epochs), "learning_rate":float(learning_rate)}
            return embedding_params_dict, {'display':'block'}
        elif embedding_method == "Feather-G":
            embedding_params_dict = {"theta_max":float(theta_max_g), "eval_points":int(eval_points_g), "order":int(order_g)}
            return embedding_params_dict, {'display':'block'}
    else:
        dash.no_update

# save clustering parameter settings
@app.callback([Output('clustering-parameters', 'data'), Output("success-parse-clustering-params", "style")], [State('n-init-kmeans', 'value'), State('max-iter-kmeans', 'value'), State('max-iter-meanshift', 'value'), State('linkage-criterion','value'), 
               State('max-iter-affinity', 'value'), State('convergence-iter-affinity','value'), State('clustering-method-selection', 'value')], Input('parse-clustering-params', 'n_clicks'), prevent_initial_call=True)
def on_click_parse_params(n_init, max_iter_kmeans, max_iter_meanshift, linkage, max_iter_affinity, convergence_iter, clustering_method, n_click):
    if n_click > 0:
        if clustering_method =='K-Means':
            clustering_params_dict = {"n_init":int(n_init), "max_iter":int(max_iter_kmeans)}
            return clustering_params_dict, {'display':'block'}
        elif clustering_method == 'Hierarchical':
            clustering_params_dict = {"linkage":linkage}
            return clustering_params_dict, {'display':'block'}
        elif clustering_method == "Mean-Shift":
            clustering_params_dict = {"max_iter":int(max_iter_meanshift)}
            return clustering_params_dict, {'display':'block'}
        elif clustering_method == "AffinityPropagation":
            clustering_params_dict = {"max_iter":int(max_iter_affinity), "convergence_iter":int(convergence_iter)}
            return clustering_params_dict, {'display':'block'}
    else:
        dash.no_update

# show silhouette plot if button clicked
@app.callback(Output("silhouette-plot", "children"), [State("ocel_obj", "data"), State("event-feature-set", "data"), State('clustering-method-selection', 'value'), State("graph-embedding-selection", "value"), State("max-clusters", "value"), State("embedding-parameters", "data") ], [Input("start-silhouette", "n_clicks")], prevent_initial_call = True)
def on_elbow_btn_click(ocel_log, selected_event_features, clustering_method, embedding_method, max_clusters, embedding_params_dict, n):
    if n > 0 and ocel_log != None:
        # load ocel
        ocel_log = pickle.loads(codecs.decode(ocel_log.encode(), "base64"))
        # extract features, get feature graphs
        feature_storage = feature_extraction.extract_features(ocel_log, selected_event_features, [], 'graph')
        # remap nodes of feature graphs
        feature_nx_graphs, attr_matrix_list = graph_embedding.feature_graphs_to_nx_graphs(feature_storage.feature_graphs)
        # embedd feature graphs
        if embedding_method == 'Graph2Vec':
            embedding = graph_embedding.perform_graph2vec(feature_nx_graphs, False, embedding_params_dict)
        elif embedding_method == 'Feather-G':
            embedding = graph_embedding.perform_feather_g(feature_nx_graphs, embedding_params_dict)
        elif embedding_method == 'AttributedGraph2Vec':
            embedding = graph_embedding.perform_attrgraph2vec(feature_nx_graphs, attr_matrix_list, embedding_params_dict)
        # calculate silhouette score for different k 
        max_clusters = int(max_clusters)
        silhouette = clustering.perform_silhouette_analysis(embedding, max_clusters, clustering_method)
        fig = silhouette_figure.create_silhouette_figure(silhouette, max_clusters, clustering_method)

        return dcc.Graph(id='silhouette-graph',figure=fig)

# perform clustering and return dataframe with process execution ids and cluster labels
@app.callback([ServersideOutput("clustered-ocels", "data"), Output("clustering-success", "children"), Output("cluster-summary-component", "children")], 
                [State("ocel_obj", "data"), State("event-feature-set", "data"), State("graph-embedding-selection", "value"), State('clustering-method-selection', 'value'), State('num-clusters-slider', 'value'), State("embedding-parameters", "data"), State('clustering-parameters', 'data')], 
                Trigger("start-clustering", "n_clicks"), memoize=True)
def on_click(ocel_log, selected_event_features, embedding_method, clustering_method, num_clusters, embedding_params_dict, clustering_params_dict, n_clicks):
    time.sleep(1)
    if n_clicks > 0:
        # load ocel
        ocel_log = pickle.loads(codecs.decode(ocel_log.encode(), "base64"))
        # extract features, get feature graphs
        feature_storage = feature_extraction.extract_features(ocel_log, selected_event_features, [], 'graph')
        # remap nodes of feature graphs
        feature_nx_graphs, attr_matrix_list = graph_embedding.feature_graphs_to_nx_graphs(feature_storage.feature_graphs)
        # embedd feature graphs
        if embedding_method == 'AttributedGraph2Vec':
            embedding = graph_embedding.perform_attrgraph2vec(feature_nx_graphs, attr_matrix_list, embedding_params_dict)
        elif embedding_method == 'Graph2Vec':
            embedding = graph_embedding.perform_graph2vec(feature_nx_graphs, False, embedding_params_dict)
        elif embedding_method == 'Feather-G':
            embedding = graph_embedding.perform_feather_g(feature_nx_graphs, embedding_params_dict)
        # cluster embedding
        if clustering_method == 'Mean-Shift':
            labels = clustering.perform_MeanShift(embedding, clustering_params_dict)
        elif clustering_method == 'K-Means':
            labels = clustering.perform_KMeans(embedding, num_clusters, clustering_params_dict)
        elif clustering_method == 'Hierarchical':
            labels = clustering.perform_HierarchicalClustering(embedding, num_clusters, clustering_params_dict)
        elif clustering_method == "AffinityPropagation":
            labels = clustering.perform_AffinityPropagation(embedding, clustering_params_dict)
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


@app.callback([Output("process-execution-graph", "children"), Output("pe-feature-table", "children")], [State("ocel_obj", "data"), State("extracted-pe-features-store", "data")], [Input("executions_dropdown", "value")], prevent_initial_call=True)
def on_extraction(ocel_log, extraction_features_list, execution_id):
    if execution_id != None:
        # load ocel
        ocel_log = pickle.loads(codecs.decode(ocel_log.encode(), "base64"))
        # get graph object of process execution
        ocel_executions_graph = process_executions.get_process_execution_graph(ocel_log, int(execution_id) - 1)
        
        # convert nx graph to dash figure 
        cyto = nxgraph_figure.create_interactive_graph(ocel_executions_graph, ocel_log)
        #fig = nxgraph_figure.create_graph_figure(ocel_executions_graph, ocel_log)

        # get DataFrame of Process Execution Features 
        if extraction_features_list != None:
            df_extr = pd.DataFrame(columns=["Feature", "Value"])
            execution_values = extraction_features_list[int(execution_id) - 1]
            df_extr["Feature"] = feature_options_extraction_renamed
            df_extr["Value"] = execution_values
            df_transposed = df_extr.T
            df_transposed.columns = df_transposed.iloc[0]
            df_transposed = df_transposed[1:]
            datatable = dbc.Table.from_dataframe(df_transposed, striped=True, bordered=True, hover=True)
        else:
            datatable = feature_table
        return  cyto, datatable
        #return dcc.Graph(id='pe-graph',figure=fig)
