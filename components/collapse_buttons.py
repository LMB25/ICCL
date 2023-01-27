import dash_bootstrap_components as dbc

# cluster evaluation
silhouette_button = dbc.Button(id="collapse-silhouette",className="fa-solid fa-chevron-right",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})
dbindex_button = dbc.Button(id="collapse-dbindex",className="fa-solid fa-chevron-right",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})
dbscan_button = dbc.Button(id="collapse-dbscan",className="fa-solid fa-chevron-right",color="dark",n_clicks=0, style={"backgroundColor": "transparent", "color": "black"})