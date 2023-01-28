import dash_bootstrap_components as dbc
from dash import html

# collapsable buttons for explanations of cluster evaluation methods
silhouette_button = html.I(id="collapse-silhouette",className="fa-solid fa-chevron-right",n_clicks=0)
dbindex_button = html.I(id="collapse-dbindex",className="fa-solid fa-chevron-right", n_clicks=0)
dbscan_button = html.I(id="collapse-dbscan",className="fa-solid fa-chevron-right",n_clicks=0)