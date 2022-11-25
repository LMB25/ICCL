import networkx as nx
from karateclub.graph_embedding import graph2vec
from karateclub import DeepWalk, Walklets
import node2vec 

def remap_nodes(graph):
    node_mapping = dict((node.event_id, i) for i, node in enumerate(graph.nodes))
    return node_mapping

def feature_graphs_to_nx_graphs(feature_graphs):
    graph_list = []
    for feature_graph in feature_graphs:
        #G = nx.DiGraph()
        G = nx.Graph()

        node_dict = remap_nodes(feature_graph)
        
        for node in feature_graph.nodes:
            G.add_node(node_dict[node.event_id])

            #for attr_key, attr_val in node.attributes.items():
            #    G.nodes[node_dict[node.event_id]][attr_key] = attr_val

        for edge in feature_graph.edges:
            G.add_edge(node_dict[edge.source], node_dict[edge.target])

        graph_list.append(G)

    return graph_list

def perform_graph2vec(graph_list, attributed):
    model = graph2vec.Graph2Vec(attributed=attributed)
    model.fit(graph_list)
    X = model.get_embedding()

    return X

def perform_node2vec(graph):
    # Generate walks
    model = node2vec.Node2Vec(graph, dimensions=2, walk_length=20, num_walks=10,workers=4)
    # Learn embeddings 
    model = model.fit(window=10, min_count=1)
    #model.wv.most_similar('1')
    model.wv.save_word2vec_format("embedding.emb") #save the embedding in file embedding.emb

def perform_deepwalk(graph):
    model = DeepWalk()
    model.fit(graph)
    embedding = model.get_embedding()
    return embedding

def perform_walklet(graph):
    model = Walklets()
    model.fit(graph)
    embedding = model.get_embedding()
    return embedding
