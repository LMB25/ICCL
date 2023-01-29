import networkx as nx
import plotly.graph_objects as go
import dash_cytoscape as cyto

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
