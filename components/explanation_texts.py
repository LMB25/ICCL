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

# Auto cluster explanation
autocluster_explanation = "tries to find the best clustering method automatically and optimizes their hyperparameters "

# DBscan analysis explanation
dbscan_explanation =  "finds core samples of high density and expands clusters from them (epsilon specifies radius of neighborhood with respect to some point, in which number of neighboring points is counted)"

# K-Means clustering explanation
kmeans_explanation = "clusters data by trying to separate samples in n groups of equal variance, centroids are chosen that minimize within-cluster sum-of-squares."

# Hierarchical clustering explanation
hierarchical_explanation = "builds nested clusters by merging or splitting them successively (this hierarchy can be represented as tree (root: unique cluster gathering all samples, leaves: clusters containing only one sample)"

# Mean-Shift clustering explanation
meanshift_explanation = "centroid based algorithm, discovers blobs in smooth density of samples (candidates for centroids are updated to be mean of the points within a given region)"

# Affinity-Propagation clustering explanation
affinity_explanation = "creates clusters by sending messages between pairs of samples until convergence, members of the data points are found that are representatives of the clusters."

# Auto Embed 
autoembed_explanation = "tries to find an optimal embedding with optimal number of dimensions with respect to the fitness/accurady of the later discovered model"

# Custom Feature Graph Embedding
cfge_explanation = "focus on features: creates FeatherNode embedding for each attributed node in feature graph and averages over these to get embedding for whole graph (structure only considered implicitly)"
#"The algorithm first creates a node embedding via FeatherNode which uses characteristic functions of node features with random walk weights to describe node neighborhoods. In the second step, the node embeddings are averaged over each dimension, resulting in a vectorized embedding of the graph."

# Graph2Vec explanation
graphvec_explanation = "embeds the whole graph based on the structure (additionaly each node can have one feature value)"
#"The algorithm first identifies subgraphs sourrounding each node in the feature graphs. By means of the Weisfeiler-Lehman’s algorithm, the subgraphs are considered as the vocabulary for a doc2vec SkipGram model. Since the graph’s structure is captured within the algorithm, feature graphs that are similar in structure will be close in the embedding space."

# Feather-G explanation
featherg_explanation = "embeds the whole graph by only focusing on the structure (features cannot be embedded)"
"The algorithm uses characteristic functions defined on graph vertices to describe the distribution of vertex attributes. Feather-G extracts node-level features that are pooled and then used to create a description of the feature graph."

# feature selection explanation
feature_selection_explanation = dbc.Card( 
                                        dbc.CardBody("Here, event features from different perspectives can be selected. You can select as many features and perspectives from the dropdown lists as you like. After selecting the features, please click the button below to parse the features."),
                                        className="mb-3",
                                        )

# list of event features and explanation
control_features_explanation = dbc.Card(
                                dbc.ListGroup(
                                    [
                                        dbc.ListGroupItem("EVENT_CURRENT_ACTIVITIES: Other current end activities (without finished events)."),
                                        dbc.ListGroupItem("EVENT_ACTIVITY: Activity that is performed in the event."),
                                        dbc.ListGroupItem("EVENT_PREVIOUS_ACTIVITY_COUNT: Number of activities that took place before the event."),
                                        dbc.ListGroupItem("EVENT_PRECEDING_ACTIVITES: Count for activities in the events before the current event.")
                                    ],
                                    flush=True,
                                ),
                            )

performance_features_explanation = dbc.Card(
                                    dbc.ListGroup(
                                    [
                                        dbc.ListGroupItem("EVENT_EXECUTION_DURATION: Duration of process execution the event belongs to."),
                                        dbc.ListGroupItem("EVENT_ELAPSED_TIME: Elapsed time from process execution start to the event."),
                                        dbc.ListGroupItem("EVENT_REMAINING_TIME: Remaining time from event to end of process execution."),
                                        dbc.ListGroupItem("EVENT_SOJOURN_TIME: Sojourn time of the event."),
                                        dbc.ListGroupItem("EVENT_WAITING_TIME: Waiting time of the event."),
                                        dbc.ListGroupItem("EVENT_DURATION: Duration of the event."),
                                    ],
                                    flush=True,
                                ),
                            )

object_features_explanation = dbc.Card(
                                dbc.ListGroup(
                                    [
                                        dbc.ListGroupItem("EVENT_PREVIOUS_OBJECT_COUNT: Number of objects involved up to the event."),
                                        dbc.ListGroupItem("EVENT_PREVIOUS_TYPE_COUNT: Count for object types before the event takes place."),
                                        dbc.ListGroupItem("EVENT_NUM_OF_OBJECTS: Number of objects involved in the event."),
                                    ],
                                    flush=True,
                                ),
                            )


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