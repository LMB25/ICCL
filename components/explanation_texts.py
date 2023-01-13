# Import necessary libraries
from dash import html, dcc
import dash_bootstrap_components as dbc

# clustering evaluation explanation
clustering_evaluation_explanation = dbc.Card( 
                                            dbc.CardBody("By clicking the button below, the OCEL will be analyzed with respect to the best clustering results. You can see the optimal number of clusters for K-Means and Hierarchical Clustering as well as the optimal parameter epsilon for DBscan clustering."),
                                            className="mb-3",
                                            )

# silhouette analysis explanation
silhouette_explanation =  dbc.CardBody("Intuitively, the silhouette score quantifies the space between different clusters. For each number of clusters, the clustering algorithm is performed. Afterwards, it is measured how similar the observation are to the assigned cluster and how dissimilar they are to the observation of the nearest cluster. The plot displays the average silhouette score for each number of clusters. The measure has the range [-1,+1], whereas a score near +1 indicates that the clusters are well separated and negative scores indicate that the samples might be wrongly separated.")

# DBindex analysis explanation
dbindex_explanation =  dbc.CardBody("The Davies–Bouldin index can intuitively be described as a measure of the ratio between within-cluster distances, and between cluster distances. The score is bounded between [0, 1]. The lower the value, the tighter the clusters and the seperation between clusters.")

# DBscan analysis explanation
dbscan_explanation =  dbc.CardBody("Density-Based Spatial Clustering of Applications with Noise is an clustering approach that finds core samples of high density and expands clusters from them. The parameter epsilon is specifying the radius of a neighborhood with respect to some point, in which the number of neighboring points is counted.")

# K-Means clustering explanation
kmeans_explanation = dbc.CardBody("The K-Means algorithm clusters data by trying to separate samples in n groups of equal variance. It aims to choose centroids that minimize the within-cluster sum-of-squares.")

# Hierarchical clustering explanation
hierarchical_explanation = dbc.CardBody("Hierarchical clustering builds nested clusters by merging or splitting them successively. This hierarchy of clusters is represented as a tree, whereas the root is the unique cluster gathering all samples and the leaves are clusters containing only one sample.")

# Mean-Shift clustering explanation
meanshift_explanation = dbc.CardBody("Mean-Shift clustering aims to discover blobs in a smooth density of samples. It is a centroid based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region")

# Affinity-Propagation clustering explanation
affinity_explanation = dbc.CardBody("Affinity-Propagation creates clusters by sending messages between pairs of samples until convergence. The algorithm finds members of the data points that are representatives of the clusters.")

# Attributed Graph2Vec explanation
attributedgraphvec_explanation = dbc.CardBody("The algorithm first creates a node embedding via FeatherNode which uses characteristic functions of node features with random walk weights to describe node neighborhoods. In the second step, the node embeddings are averaged over each dimension, resulting in a vectorized embedding of the graph.")

# Graph2Vec explanation
graphvec_explanation = dbc.CardBody("The algorithm first identifies subgraphs sourrounding each node in the feature graphs. By means of the Weisfeiler-Lehman’s algorithm, the subgraphs are considered as the vocabulary for a doc2vec SkipGram model. Since the graph’s structure is captured within the algorithm, feature graphs that are similar in structure will be close in the embedding space.")

# Feather-G explanation
featherg_explanation = dbc.CardBody("The algorithm uses characteristic functions defined on graph vertices to describe the distribution of vertex attributes. Feather-G extracts node-level features that are pooled and then used to create a description of the feature graph.")

# list of features and explanation
features_explanation = dbc.Card(
                                dbc.ListGroup(
                                    [
                                        dbc.ListGroupItem("EVENT_REMAINING_TIME: Remaining time from event to end of process execution."),
                                        dbc.ListGroupItem("EVENT_ELAPSED_TIME: Elapsed time from process execution start to the event."),
                                        dbc.ListGroupItem("EVENT_ACTIVITY: Activity that is performed in the event."),
                                        dbc.ListGroupItem("EVENT_NUM_OF_OBJECTS: Number of objects involved in the event."),
                                        dbc.ListGroupItem("EVENT_PREVIOUS_OBJECT_COUNT: Number of objects involved up to the event."),
                                        dbc.ListGroupItem("EVENT_PREVIOUS_ACTIVITY_COUNT: Number of activities that took place before the event."),
                                        dbc.ListGroupItem("EVENT_DURATION: Duration of the event."),
                                    ],
                                    flush=True,
                                ),
                            )


# process executions explanation
process_executions_explanation = dbc.Card(
                                dbc.CardBody("Hier sollte eine Erklärung zu den Process Executions stehen. Was zeigt der Plot des Process Execution Graphs an?"),
                                className="mb-3",
                                )


# explanation for process execution extraction type selection
extraction_type_explanation = dbc.Card(
                                dbc.ListGroup(
                                    [
                                        dbc.ListGroupItem("Connected Components: process executions are extracted based on the connected components of the object graph. All transitively connected objects form one process execution."),
                                        dbc.ListGroupItem("Leading Object Type: process execution is constructed for each object of a chosen leading object type. Connected objects are added to this process execution unless a connected object of the same type has a lower distance to the leading object."),
                                    ],
                                    flush=True,
                                ),
                            )