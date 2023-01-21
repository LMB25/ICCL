from dash import html
import dash_bootstrap_components as dbc
import networkx as nx
import plotly.graph_objects as go
from ocpa.algo.predictive_monitoring.event_based_features.extraction_functions import event_activity
import dash_cytoscape as cyto

# process execution displayed as Plotly scatterplot
def create_graph_figure(G, ocel):
    pos = nx.layout.spectral_layout(G)#spring_layout(G)
    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])
        
    edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5,color='#888'),
    hoverinfo='none',
    mode='lines')
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    textposition="top center",
    mode='markers+text',
    hoverinfo='text',
    marker=dict(line=dict(width=2)))
    
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        
    node_text = []
    for node in G.nodes:
        node_text.append(ocel.get_value(node, "event_activity"))

    node_trace.text = node_text


    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Graph of Process Execution',
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )

    return fig

# process execution displayed as cytoscape figure
def create_interactive_graph(G, ocel):
    pos = nx.layout.spectral_layout(G,scale=200)
    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])
    
    nodes = [
        {
            'data': {'id': str(node), 'label': ocel.get_value(node, "event_activity")},
            'position': {'x': 5 * G.nodes[node]['pos'][0], 'y': G.nodes[node]['pos'][1]}
        }
        for node in G.nodes()
    ]

    edges = [
        {'data': {'source': str(edge[0]), 'target': str(edge[1])}}
        for edge in G.edges()
    ]

    elements = nodes + edges
    
    return cyto.Cytoscape(
        id='cytoscape-layout-1',
        elements=elements,
        style={'width': '100%', 'height': '350px'},
        layout={
            'name': 'preset'
        }
    )
