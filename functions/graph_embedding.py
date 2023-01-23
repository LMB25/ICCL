import networkx as nx
from karateclub.graph_embedding import graph2vec, feathergraph
from karateclub import DeepWalk, Walklets, TENE, FeatherNode
import node2vec 
import numpy as np

from sklearn.preprocessing import normalize
from kneed import KneeLocator

def remap_nodes(graph):
    node_mapping = dict((node.event_id, i) for i, node in enumerate(graph.nodes))
    return node_mapping

def feature_graphs_to_nx_graphs(feature_graphs):
    graph_list = []
    attr_matrix_list = []
    for feature_graph in feature_graphs:
        #G = nx.DiGraph()
        G = nx.Graph()

        node_dict = remap_nodes(feature_graph)
        
        attr_for_graphs = []
        for node in feature_graph.nodes:
            G.add_node(node_dict[node.event_id])
            attr_for_graphs.append(list(node.attributes.values()))
   
        for edge in feature_graph.edges:
            G.add_edge(node_dict[edge.source], node_dict[edge.target])

        graph_list.append(G)
        attr_matrix_list.append(np.array(attr_for_graphs))
    
    return graph_list, attr_matrix_list

def find_optimal_dim(feature_nx_graphs, attr_matrix_list):
    #check for the biggest graph 
    i_largest_graph = 0
    for i, graph in enumerate(feature_nx_graphs):
        if graph.number_of_nodes() > feature_nx_graphs[i_largest_graph].number_of_nodes():
            i_largest_graph = i
    
    #the maximal possible dimension size is the number of nodes of the biggest graph
    max_dim = feature_nx_graphs[i_largest_graph].number_of_nodes()
    
    ref_emb = perform_feather_node(feature_nx_graphs[i_largest_graph], attr_matrix_list[i_largest_graph], max_dim)
    Va = normalize(np.array(ref_emb))
    CVa = Va @ (Va.T)
    N = CVa.shape[0] 
    
    #now we try out smaller dimensions and check the loss (w.r.t. the largest dimension size)
    losses = []
    for dim in range(max_dim-1, 0, -1):
        emb = perform_feather_node(feature_nx_graphs[i_largest_graph], attr_matrix_list[i_largest_graph], dim)
        Vb = normalize(np.array(emb))
        CVb = Vb @ (Vb.T)
        
        #compute normalized embedding loss                
        Loss = 0
        for i in range(0,CVa.shape[0]):
            for j in range(0,CVa.shape[1]):
                if i<j:
                    Loss += abs(CVa[j,i] - CVb[j,i])    
        Loss = (2/(N*(N-1))) * Loss
                        
        losses.append(Loss)

    #try to find the ellbow point, i.e. the smallest dimension size that still has good performance
    #try:
        #kn = KneeLocator(range(1,max_dim),losses, curve='convex', direction='increasing', interp_method='interp1d')
        #opt_dim = kn.knee
    #except:
    opt_dim = max_dim 
        
    return opt_dim

def perform_feather_node(graph, attr_matrix, num_dim):
    model = FeatherNode(reduction_dimensions=num_dim)
    model.fit(graph, attr_matrix)
    return model.get_embedding()

def perform_cfge(graph_list, attr_matrix_list, embedding_params):
    X_graphs = []
    for graph, attr_matrix in zip(graph_list, attr_matrix_list):
        #model = TENE(dimensions=100)
        model = FeatherNode(reduction_dimensions=embedding_params['svd_dimensions'], svd_iterations=embedding_params['svd_iterations'], theta_max=embedding_params['theta_max'], eval_points=embedding_params['eval_points'], order=embedding_params['order'])
                
        model.fit(graph, attr_matrix)
        X = model.get_embedding()
        
        X_graphs.append(X.mean(axis=0))
        
        #for each node add the mean of the node embedding (stored in X) as the feature attribute in the original graph
        # for i, attr in enumerate(X):
        #     feature_value = np.mean(attr)
        #     graph.nodes[i]["feature"] = str(feature_value)[0:3]
    
    X_graphs = np.array(X_graphs)         
         
    #print(X_graphs)
       
    return X_graphs#perform_graph2vec(graph_list, attributed=True) 

def perform_graph2vec(graph_list, attributed, embedding_params):
    model = graph2vec.Graph2Vec(attributed=attributed, wl_iterations=embedding_params['wl_iterations'], dimensions=embedding_params['dimensions'],epochs=embedding_params['epochs'],learning_rate=embedding_params['learning_rate'], min_count=1)
    model.fit(graph_list)
    X = model.get_embedding()

    return X

def perform_feather_g(graph_list, embedding_params):
    model = feathergraph.FeatherGraph(order=embedding_params['order'], eval_points=embedding_params['eval_points'], theta_max=embedding_params['theta_max'])
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