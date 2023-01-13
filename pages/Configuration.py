# Import necessary libraries 
from dash import html, dcc, ctx
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash
import time
from app import app
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Dash, Trigger, ServersideOutput
from functions import process_executions, feature_extraction, graph_embedding, clustering, dataimport
from components import nxgraph_figure, silhouette_figure, input_forms, explanation_texts, collapse_buttons
import pickle
import codecs
import pandas as pd


# event feature set store
event_store = dcc.Store('event-feature-set', storage_type='local')
# graph embedding parameters store
embedding_params_store = dcc.Store('embedding-parameters', storage_type='local')
# Define Store object for List of list of Process Execution Features
extracted_pe_features = dcc.Store(id='extracted-pe-features-store')
# clustering parameters store
clustering_params_store = dcc.Store('clustering-parameters', storage_type='local')

# options for event based feature selection
feature_options_event = ['EVENT_REMAINING_TIME', 'EVENT_ELAPSED_TIME', 'EVENT_ACTIVITY', 'EVENT_NUM_OF_OBJECTS', 'EVENT_PREVIOUS_OBJECT_COUNT', 'EVENT_PREVIOUS_ACTIVITY_COUNT', 'EVENT_DURATION']
                        
# event based feature selection dropdown
event_feature_selection_dropdown= dcc.Dropdown(id='feature-selection-event', options=[{'label': i, 'value': i} for i in feature_options_event], multi=True, value=[])#feature_options_event)

# empty DataTable for Process Execution Features
feature_options_extraction_renamed = ["Number of Events", "Number of Ending Events", "Throughput Duration", "Number of Objects", "Unique Activities", "Number of Starting Events", "Duration of Last Event"]
dummy_df = pd.DataFrame(columns=feature_options_extraction_renamed)
feature_table = dbc.Table.from_dataframe(dummy_df, striped=True, bordered=True, hover=True)


# create empty div for embedding param form
embedding_param_form = html.Div([input_forms.embedding_params_form_attributed, input_forms.embedding_params_form_graph2vec, input_forms.embedding_params_form_featherg], id='embedding-params-div', style={'display': 'block'})

# create empty div for clustering param form
clustering_param_form = html.Div([input_forms.clustering_params_form_kmeans, input_forms.clustering_params_form_hierarchical, input_forms.clustering_params_form_meanshift, input_forms.clustering_params_form_affinity, input_forms.clustering_params_form_dbscan], id='clustering-params-div', style={'display':'block'})

# Define the page layout
layout = dbc.Tabs([
        event_store, embedding_params_store, extracted_pe_features, clustering_params_store, 
        dbc.Tab([
                html.Br(),
                html.H5("Feature Explanation:"),
                dbc.Row([dbc.Col([explanation_texts.features_explanation], width=7),]),
                html.Hr(),
                dbc.Row(dbc.Col([html.H5("Feature Selection:"), html.Div("Select Event Features for Clustering:")], width=7)),
                dbc.Row([dbc.Col([html.Div(event_feature_selection_dropdown)], width=7), dbc.Col([dbc.Button("Set Selected Features", className="me-2", id='set-features', n_clicks=0), html.Div(id='feature-sucess')], width=5)])
        ], label="Features", tab_id='features-tab', label_style={'background-color': '#8daed9'}),
        dbc.Tab([
                html.Br(),
                dbc.Row([
                    dbc.Col([html.H5("Select Graph Embedding Method")], width=5),
                    dbc.Col(["Attributed Graph2Vec", collapse_buttons.attr_graph_vec_button]),
                    dbc.Col(["Graph2Vec", collapse_buttons.graph_vec_button]),
                    dbc.Col(["Feather-G", collapse_buttons.featherg_button]),
                ]),
                dbc.Row([
                    dbc.Col([html.Div(dbc.RadioItems(options=[{"label": "AttributedGraph2Vec", "value": "AttributedGraph2Vec"},{"label": "Graph2Vec", "value": 'Graph2Vec'},{"label": "Feather-G", "value": "Feather-G"}], value="AttributedGraph2Vec", id="graph-embedding-selection"),)], width=5),
                    dbc.Col([dbc.Collapse(dbc.Card(children=[],id='embedding-explanations'),id="embedding-collapse", is_open=False)])
                ]),
                html.Br(),
                dbc.Row([ 
                        dbc.Col([embedding_param_form], width=8),
                        dbc.Col([dbc.Button("Parse Embedding Parameters", id="parse-embedding-params", className="me-2", n_clicks=0)]),
                        html.Div("Parameters successfully parsed.", style={'display':'none'}, id='success-parse-embedding-params'),
                        ],
                        style={'display':'block'}),
                html.Br(),
                ], label="Embedding", tab_id='embedding-tab', label_style={'background-color': '#8dd996'}),
        dbc.Tab([
                html.Br(),
                dbc.Row([
                        dbc.Col([html.H5("Select Clustering Technique")],width=4),
                        dbc.Col(["K-Means", collapse_buttons.kmeans_button]),
                        dbc.Col(["Hierarchical", collapse_buttons.hierarchical_button]),
                        dbc.Col(["Mean-Shift", collapse_buttons.meanshift_button]),
                        dbc.Col(["Affinity-Prop.", collapse_buttons.affinity_button]),
                        dbc.Col(["DBscan", collapse_buttons.dbscan_expl_button]),
                        ]),
                dbc.Row([
                        dbc.Col([html.Div(dbc.RadioItems(options=[{"label": "K-Means", "value": "K-Means"},{"label": "Hierarchical", "value": "Hierarchical"},{"label": "Mean-Shift", "value": 'Mean-Shift'},{"label": "Affinity-Propagation", "value":"AffinityPropagation"}, {'label':'DBscan', 'value':'DBscan'}], value="K-Means", id="clustering-method-selection"),)], width=4),
                        dbc.Col([dbc.Collapse(dbc.Card(children=[],id='clustering-explanations'),id="clustering-collapse", is_open=False)])
                        ]),
                html.Br(),
                dbc.Row([dbc.Col([html.H5("Modify Clustering Embedding Parameters:")])]),
                dbc.Row([
                        dbc.Col([html.Div("Select Number of Clusters"), html.Div(dcc.Slider(1,14,1, value=2, id='num-clusters-slider', disabled=True)), clustering_param_form], width=7),
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
                        dbc.Col(explanation_texts.clustering_evaluation_explanation)
                        ]),
                dbc.Row([
                        dbc.Col(["Davies-Bouldin index", collapse_buttons.dbindex_button]),
                        dbc.Col(["Silhouette", collapse_buttons.silhouette_button]),
                        dbc.Col(["DBscan", collapse_buttons.dbscan_button]),
                        ]),
                dbc.Row([
                        dbc.Collapse(dbc.Card(children=[],id='evaluation-explanations'),id="evaluation-collapse", is_open=False)
                        ]),
                html.Br(),
                html.Div([
                          dbc.Col([dbc.Alert([html.I(className="fa-solid fa-triangle-exclamation"),"You have to parse embedding parameters first."],color="warning", className="d-flex align-items-center")], width=4)
                         ], id='alert-params-evaluation', hidden=False),
                dbc.Row([
                        dbc.Col([dbc.Button("Analyze Clustering Techniques", className="me-1", id='evaluation-button', n_clicks=0, disabled=True)])
                        ]),
                html.Br(),
                html.Div([
                html.H5("Hierarchical Clustering"),
                html.Div("Ward Linkage"),
                dbc.Row([
                        dbc.Col([html.Img(id='ward-evaluation')])
                        ]),
                html.Div("Average Linkage"), 
                dbc.Row([
                        dbc.Col([html.Img(id='average-evaluation')])
                        ]),
                html.Br(),
                html.H5("K-Means Clustering"),
                dbc.Row([
                        dbc.Col([html.Img(id='kmeans-evaluation')])
                        ]),
                html.Br(),
                html.H5("DBscan Clustering"),
                dbc.Row([
                        dbc.Col([html.Img(id='dbscan-evaluation')])
                        ]),
                        ], id='cluster-evaluation-result', hidden=True),
                ], label='Cluster Evaluation', tab_id='cluster-eval'),
        dbc.Tab([
                html.Br(),
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
        ], id='configuration-tabs',active_tab='features-tab')



@app.callback([Output('ward-evaluation', 'src'), Output('average-evaluation', 'src'), Output('kmeans-evaluation', 'src'), Output('dbscan-evaluation', 'src'), Output('cluster-evaluation-result', 'hidden')], 
               Input('evaluation-button', 'n_clicks'), [State("ocel_obj", "data"), State("event-feature-set", "data"), State("graph-embedding-selection", "value"), State("embedding-parameters", "data") ], prevent_initial_call=True)
def on_button_click(n_clicks, ocel_log, selected_event_features, embedding_method, embedding_params_dict):
    if n_clicks > 0:
        # load ocel
        ocel_log = pickle.loads(codecs.decode(ocel_log.encode(), "base64"))
        # extract features, get feature graphs
        feature_storage = feature_extraction.extract_features(ocel_log, selected_event_features, [], 'graph')
        # remap nodes of feature graphs
        feature_nx_graphs, attr_matrix_list = graph_embedding.feature_graphs_to_nx_graphs(feature_storage.feature_graphs)
        # embedd feature graphs
        if embedding_method == 'Graph2Vec':
            X = graph_embedding.perform_graph2vec(feature_nx_graphs, False, embedding_params_dict)
        elif embedding_method == 'Feather-G':
            X = graph_embedding.perform_feather_g(feature_nx_graphs, embedding_params_dict)
        elif embedding_method == 'AttributedGraph2Vec':
            X = graph_embedding.perform_attrgraph2vec(feature_nx_graphs, attr_matrix_list, embedding_params_dict)

        try:
            hierarchical_ward = clustering.cluster_evaluation_hierarchical(X, "ward")
        except: 
            hierarchical_ward = None 
        try:
            hierarchical_average = clustering.cluster_evaluation_hierarchical(X, "average")
        except:    
            hierarchical_average = None
        try:
            kmeans = clustering.cluster_evaluation_kmeans(X)
        except:
            kmeans = None
        try:
            dbscan = clustering.cluster_evaluation_dbscan(X)
        except:
            dbscan = None

        return hierarchical_ward, hierarchical_average, kmeans, dbscan, False


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

# extract process execution features, get list of lists of features
@app.callback([ServersideOutput('extracted-pe-features-store', 'data'), Output("pe-feature-success", "children")], Trigger("start-feature-extraction-pe", "n_clicks"), State("ocel_obj", "data"), prevent_initial_call=True, memoize=True)
def on_click(n_clicks, ocel):
    if n_clicks > 0:
        # load ocel
        ocel = pickle.loads(codecs.decode(ocel.encode(), "base64"))
        list_features = feature_extraction.create_extraction_features(ocel)
        return list_features, "Features successfully extracted."
    else:
        raise PreventUpdate


# show alert if cluster evaluation is selected, but no embedding parameters are parsed
# if embedding parameters are parsed, hide alert and enable button
@app.callback([Output("alert-params-evaluation","hidden"), Output("evaluation-button", "disabled")], Input("embedding-parameters","data"), prevent_initial_call=True)
def on_embedding_parameters_parsed(embedding_data):
    if embedding_data != None:
        return True, False
    else:
        return False, True

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
@app.callback([Output("clustering-params-div-kmeans", "style"), Output("clustering-params-div-hierarchical", "style"), Output("clustering-params-div-meanshift", "style"), Output("clustering-params-div-affinity", "style"), 
               Output("clustering-params-div-dbscan", "style")], Input("clustering-method-selection", "value"))
def on_clustering_selection(clustering_method):
    if clustering_method == 'K-Means':
        return {'display':'block'}, {'display':'none'}, {'display':'none'}, {'display':'none'}, {'display':'none'}
    elif clustering_method == 'Hierarchical':
        return {'display':'none'}, {'display':'block'}, {'display':'none'}, {'display':'none'}, {'display':'none'}
    elif clustering_method == "Mean-Shift":
        return {'display':'none'}, {'display':'none'}, {'display':'block'}, {'display':'none'}, {'display':'none'}
    elif clustering_method == 'AffinityPropagation':
        return {'display':'none'}, {'display':'none'}, {'display':'none'}, {'display':'block'}, {'display':'none'}
    elif clustering_method == 'DBscan':
        return {'display':'none'}, {'display':'none'}, {'display':'none'}, {'display':'none'}, {'display':'block'}
    else:
        return {'display':'none'}, {'display':'none'}, {'display':'none'}, {'display':'none'}, {'display':'none'}

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

# enable number of clusters slider, if kmeans or hierarchical clustering is selected
@app.callback(Output('num-clusters-slider', 'disabled'), Input('clustering-method-selection', 'value'))
def on_clustering_selection(clustering_method):
    if (clustering_method == 'K-Means') or (clustering_method == 'Hierarchical'):
        return False 
    else:
        return True

# save clustering parameter settings
@app.callback([Output('clustering-parameters', 'data'), Output("success-parse-clustering-params", "style")], [State('n-init-kmeans', 'value'), State('max-iter-kmeans', 'value'), State('max-iter-meanshift', 'value'), State('linkage-criterion','value'), 
               State('max-iter-affinity', 'value'), State('convergence-iter-affinity','value'), State('epsilon', 'value'), State('min-samples', 'value'), State('clustering-method-selection', 'value')], Input('parse-clustering-params', 'n_clicks'), prevent_initial_call=True)
def on_click_parse_params(n_init, max_iter_kmeans, max_iter_meanshift, linkage, max_iter_affinity, convergence_iter, eps, min_samples, clustering_method, n_click):
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
        elif clustering_method == "DBscan":
            clustering_params_dict = {"eps":float(eps), "min_samples":int(min_samples)}
            return clustering_params_dict, {'display':'block'}
    else:
        dash.no_update


# perform clustering and return dataframe with process execution ids and cluster labels
@app.callback([ServersideOutput("clustered-ocels", "data"), ServersideOutput("extracted-pe-features-cluster-store", "data"), Output("clustering-success", "children"), Output("cluster-summary-component", "children")], 
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
        print("Embedding worked.")
        # cluster embedding
        if clustering_method == 'Mean-Shift':
            labels = clustering.perform_MeanShift(embedding, clustering_params_dict)
        elif clustering_method == 'K-Means':
            labels = clustering.perform_KMeans(embedding, num_clusters, clustering_params_dict)
        elif clustering_method == 'Hierarchical':
            labels = clustering.perform_HierarchicalClustering(embedding, num_clusters, clustering_params_dict)
        elif clustering_method == "AffinityPropagation":
            labels = clustering.perform_AffinityPropagation(embedding, clustering_params_dict)
        elif clustering_method == "DBscan":
            labels = clustering.perform_DBSCAN(embedding, clustering_params_dict)
        # create Dataframe with process execution id and cluster labels
        clustered_df = clustering.create_clustered_df(ocel_log.process_executions, labels)
        # get summary of clusters
        cluster_summary_df = clustering.get_cluster_summary(clustered_df)
        # partition ocel into clustered ocels
        sub_ocels = clustering.partition_ocel(ocel_log, clustered_df)
        # get average process execution features for each cluster
        average_pe_features = feature_extraction.create_cluster_feature_summary(sub_ocels)
        # encoding/ storing of sub ocels
        sub_ocels_encoded = [codecs.encode(pickle.dumps(ocel), "base64").decode() for ocel in sub_ocels]
    else:
        raise PreventUpdate
    return sub_ocels_encoded, average_pe_features, "Clustering successfully performed.", dbc.Table.from_dataframe(cluster_summary_df, striped=True, bordered=True, hover=True, id="cluster-summary-table")

@app.callback([Output("process-executions-summary", "children"), Output("executions_dropdown", "options")], [Input("execution-store", "data")])
def on_extraction(executions):
    if executions is None: 
        return dash.no_update
    else:
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


# collapse evaluation explanation
@app.callback(
    [Output("evaluation-collapse", "is_open"), Output("evaluation-explanations","children")],
    [Input("collapse-silhouette", "n_clicks"), Input("collapse-dbindex", "n_clicks"), Input("collapse-dbscan", "n_clicks")],
    [State("evaluation-collapse", "is_open")], prevent_initial_call=True
)
def show_explanation(silhouette_n, dbindex_n, dbscan_n, is_open):
    triggered_id = ctx.triggered_id
    if (silhouette_n > 0) or (dbindex_n > 0) or (dbscan_n > 0):
        if triggered_id == "collapse-silhouette":
            explanation_text = explanation_texts.silhouette_explanation
            if silhouette_n % 2 != 0:
                open_status = True 
            else:
                open_status = False 
        elif triggered_id == "collapse-dbindex":
            explanation_text = explanation_texts.dbindex_explanation
            if dbindex_n % 2 != 0:
                open_status = True 
            else:
                open_status = False 
        elif triggered_id == "collapse-dbscan":
            explanation_text = explanation_texts.dbscan_explanation
            if dbscan_n % 2 != 0:
                open_status = True 
            else:
                open_status = False 
        return open_status, explanation_text
    return is_open, []

# collapse clustering explanation
@app.callback(
    [Output("clustering-collapse", "is_open"), Output("clustering-explanations","children")],
    [Input("collapse-kmeans", "n_clicks"), Input("collapse-dbscan-expl", "n_clicks"), Input("collapse-affinity", "n_clicks"), Input("collapse-meanshift", "n_clicks"), Input("collapse-hierarchical", "n_clicks")],
    [State("clustering-collapse", "is_open")], prevent_initial_call=True
)
def show_explanation(kmeans_n, dbscan_n, affinity_n, meanshift_n, hierarchical_n, is_open):
    triggered_id = ctx.triggered_id
    if (kmeans_n > 0) or (dbscan_n > 0) or (affinity_n > 0) or (meanshift_n > 0) or (hierarchical_n > 0):
        if triggered_id == "collapse-kmeans":
            explanation_text = explanation_texts.kmeans_explanation
            if kmeans_n % 2 != 0:
                open_status = True 
            else:
                open_status = False 
        elif triggered_id == "collapse-dbscan-expl":
            explanation_text = explanation_texts.dbscan_explanation
            if dbscan_n % 2 != 0:
                open_status = True 
            else:
                open_status = False 
        elif triggered_id == "collapse-affinity":
            explanation_text = explanation_texts.affinity_explanation
            if affinity_n % 2 != 0:
                open_status = True 
            else:
                open_status = False 
        elif triggered_id == "collapse-meanshift":
            explanation_text = explanation_texts.meanshift_explanation
            if meanshift_n % 2 != 0:
                open_status = True 
            else:
                open_status = False 
        elif triggered_id == "collapse-hierarchical":
            explanation_text = explanation_texts.hierarchical_explanation
            if hierarchical_n % 2 != 0:
                open_status = True 
            else:
                open_status = False 
        return open_status, explanation_text
    return is_open, []

# collapse embedding explanation
@app.callback(
    [Output("embedding-collapse", "is_open"), Output("embedding-explanations","children")],
    [Input("collapse-attr-graph-vec", "n_clicks"), Input("collapse-graph-vec", "n_clicks"), Input("collapse-feather-g", "n_clicks")],
    [State("embedding-collapse", "is_open")], prevent_initial_call=True
)
def show_explanation(attr_graph_vec_n, graph_vec_n, featherg_n, is_open):
    triggered_id = ctx.triggered_id
    if (attr_graph_vec_n > 0) or (graph_vec_n > 0) or (featherg_n > 0):
        if triggered_id == "collapse-attr-graph-vec":
            explanation_text = explanation_texts.attributedgraphvec_explanation
            if attr_graph_vec_n % 2 != 0:
                open_status = True 
            else:
                open_status = False 
        elif triggered_id == "collapse-graph-vec":
            explanation_text = explanation_texts.graphvec_explanation
            if graph_vec_n % 2 != 0:
                open_status = True 
            else:
                open_status = False 
        elif triggered_id == "collapse-feather-g":
            explanation_text = explanation_texts.featherg_explanation
            if featherg_n % 2 != 0:
                open_status = True 
            else:
                open_status = False 
        return open_status, explanation_text
    return is_open, []
