import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import html
import pandas as pd

def create_silhouette_figure(silhouette, max_clusters, method):

    x_range = [i for i in range(2,max_clusters+1)]

    fig = go.Figure(
                    data=go.Scatter(x=x_range, y=silhouette)
                    )

    fig.update_layout(
                        title= method + " Silhouette Analysis",
                        xaxis_title="Number of Clusters",
                        yaxis_title="Silhouette Score",
                     )

    return fig
