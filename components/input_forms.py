# Import necessary libraries
from dash import html, dcc
import dash_bootstrap_components as dbc

# create form for csv parameter input
csv_import = html.Div([
        html.H5("Please specify the necessary parameters for OCEL csv import"),
        dbc.Row([
            dbc.Col(html.P("Select event id column: ")),
            dbc.Col(dcc.Dropdown(id='id_name'))
        ]),
        dbc.Row([
            dbc.Col(html.P("Select timestamp column: ")),
            dbc.Col(dcc.Dropdown(id='time_name'))
        ]),
        dbc.Row([
            dbc.Col(html.P("Select event activity column: ")),
            dbc.Col(dcc.Dropdown(id='act_name'))
        ]),
        dbc.Row([
            dbc.Col(html.P("Enter object names: ")),
            dbc.Col(dcc.Dropdown(id='obj_names', multi=True))
        ]),
        html.Br(),
        dbc.Row([
            dbc.Button("Parse csv Parameters", color="warning", id="parse-csv", className="me-2", n_clicks=0)
        ])], id='csv-import', style={'display': 'none'})

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
