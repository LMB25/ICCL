from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash_extensions.enrich import ServersideOutput
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from app import app, long_callback_manager
import time
import pickle, codecs
from functions import feature_extraction, graph_embedding, clustering, dataimport


layout = dbc.Container([
    dcc.Store(id='logs'),
    html.Center(html.H1("Automatic OCEL Simplification")),
    html.Div([
        dbc.Card(dbc.CardBody("With this mode the whole workflow of feature extraction, feature encoding and clustering is done automatically, where the optimal parameters for the graph embedding and clsutering are automatically chosen by performing hyperparameter tuning."))
    ]),
    dbc.Button("Start Automated Clustering", className="me-2", id='start-auto-clustering', n_clicks=0),
    dbc.Button("Cancel", className="me-2", id='cancel-auto-clustering', n_clicks=0),
    dbc.Row(html.Progress(id="progress-bar", value="0")),
    dbc.Row(html.Div(id="progress-message")),
    html.Div(id="auto-cluster-summary-component"),
    html.Div(dbc.Button("Discover Process Models", className="me-2", n_clicks=0, id="discover-auto-button"), hidden=True, id='discover-auto')
])

#perform the automated feature extraction, graph embedding and clustering
@app.long_callback(prevent_initial_call=True, output=(
    ServersideOutput("clustered-ocels", "data"), 
    ServersideOutput("extracted-pe-features-cluster-store", "data"), 
    Output("auto-cluster-summary-component", "children"),
    Output("discover-auto", "hidden")
    ), 
              inputs=(State("ocel_obj", "data"), Input("start-auto-clustering", "n_clicks")),
              running=[
                  (Output("start-auto-clustering", "disabled"), True, False),
                  (Output("cancel-auto-clustering", "disabled"), False, True),
                  (Output("progress-bar", "style"),{"visibility": "visible"},{"visibility": "hidden"}),
                  ],
              cancel=[Input("cancel-auto-clustering", "n_clicks")],
              progress=[Output("progress-bar", "value"), Output("progress-bar", "max"), Output("progress-message","children")]
              )
def on_click(set_progress, ocel_log, n_clicks):
    if n_clicks>0:        
        #Feature Extraction
        set_progress(("0","10","... Extracting Features"))
            #simply choose all events
        selected_event_features=['EVENT_REMAINING_TIME', 'EVENT_ELAPSED_TIME', 'EVENT_ACTIVITY', 'EVENT_NUM_OF_OBJECTS', 'EVENT_PREVIOUS_OBJECT_COUNT', 'EVENT_PREVIOUS_ACTIVITY_COUNT', 'EVENT_DURATION']
        
        ocel_log = pickle.loads(codecs.decode(ocel_log.encode(), "base64"))
        feature_storage = feature_extraction.extract_features(ocel_log, selected_event_features, [], 'graph')
        
        
        #Graph Embedding     
        set_progress(("3","10","... Embedding Features"))   
        feature_nx_graphs, attr_matrix_list = graph_embedding.feature_graphs_to_nx_graphs(feature_storage.feature_graphs)
        embedding_params_dict = {"svd_dimensions":int(64), "svd_iterations":int(20), "theta_max":float(2.5), "eval_points":int(25), "order":int(5)}
        embedding = graph_embedding.perform_attrgraph2vec(feature_nx_graphs, attr_matrix_list, embedding_params_dict)
                
        
        
        #Clustering 
        set_progress(("6","10","... Perform Clustering"))      
        clustering_params_dict = {"max_iter":int(300)}
        labels = clustering.perform_MeanShift(embedding, clustering_params_dict)
        
        set_progress(("8","10","... Partition OCEL"))      
        # create Dataframe with process execution id and cluster labels
        clustered_df = clustering.create_clustered_df(ocel_log.process_executions, labels)
        # get summary of clusters
        cluster_summary_df = clustering.get_cluster_summary(clustered_df)
        # partition ocel into clustered ocels
        ocel_df, _ = dataimport.ocel_to_df_params(ocel_log)
        sub_ocels = clustering.partition_ocel(ocel_log, ocel_df, clustered_df)
        # get average process execution features for each cluster
        average_pe_features = feature_extraction.create_cluster_feature_summary(sub_ocels)
        # encoding/ storing of sub ocels
        sub_ocels_encoded = [codecs.encode(pickle.dumps(ocel), "base64").decode() for ocel in sub_ocels]
        
        set_progress(("10","10","Done!"))      
        
        return sub_ocels_encoded, average_pe_features, dbc.Table.from_dataframe(cluster_summary_df, striped=True, bordered=True, hover=True, id="cluster-summary-table"), False
    
    else:
        raise PreventUpdate
    
#after clustering has been performed, user can directly be forwarded to discovery page
@app.callback(Output('url', 'pathname'), Input("discover-auto-button","n_clicks"))  
def forward_to_discovery(n_clicks):
    if n_clicks>0:
        return "/page-3/2"
    else:
        raise PreventUpdate
    