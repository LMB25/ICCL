import dash_bootstrap_components as dbc

# cluster evaluation
silhouette_button = dbc.Button(id="collapse-silhouette",className="fas fa-chevron-right me-3",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})
dbindex_button = dbc.Button(id="collapse-dbindex",className="fas fa-chevron-right me-3",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})
derivative_button = dbc.Button(id="collapse-derivative",className="fas fa-chevron-right me-3",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})
dbscan_button = dbc.Button(id="collapse-dbscan",className="fas fa-chevron-right me-3",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})