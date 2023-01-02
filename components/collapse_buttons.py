import dash_bootstrap_components as dbc

# cluster evaluation
silhouette_button = dbc.Button(id="collapse-silhouette",className="fas fa-chevron-right me-3",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})
dbindex_button = dbc.Button(id="collapse-dbindex",className="fas fa-chevron-right me-3",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})
derivative_button = dbc.Button(id="collapse-derivative",className="fas fa-chevron-right me-3",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})
dbscan_button = dbc.Button(id="collapse-dbscan",className="fas fa-chevron-right me-3",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})

# cluster algorithms
kmeans_button = dbc.Button(id="collapse-kmeans",className="fas fa-chevron-right me-3",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})
hierarchical_button = dbc.Button(id="collapse-hierarchical",className="fas fa-chevron-right me-3",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})
meanshift_button = dbc.Button(id="collapse-meanshift",className="fas fa-chevron-right me-3",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})
affinity_button = dbc.Button(id="collapse-affinity",className="fas fa-chevron-right me-3",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})
dbscan_expl_button = dbc.Button(id="collapse-dbscan-expl",className="fas fa-chevron-right me-3",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})

# embedding algorithms
attr_graph_vec_button = dbc.Button(id="collapse-attr-graph-vec",className="fas fa-chevron-right me-3",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})
graph_vec_button = dbc.Button(id="collapse-graph-vec",className="fas fa-chevron-right me-3",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})
featherg_button = dbc.Button(id="collapse-feather-g",className="fas fa-chevron-right me-3",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})