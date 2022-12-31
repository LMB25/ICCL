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

# Derivative analysis explanation
derivative_explanation =  dbc.CardBody("The derivative method compares each cluster merge’s height to the average mean and normalizes it by the standard deviation formed over the depth previous levels.")

# DBscan analysis explanation
dbscan_explanation =  dbc.CardBody("Density-Based Spatial Clustering of Applications with Noise is an clustering approach that finds core samples of high density and expands clusters from them. The parameter epsilon is specifying the radius of a neighborhood with respect to some point, in which the number of neighboring points is counted.")


# list of features and explanation
features_explanation = dbc.Card(
                                dbc.ListGroup(
                                    [
                                        dbc.ListGroupItem("EVENT_REMAINING_TIME: Remaining time from event to end of process execution."),
                                        dbc.ListGroupItem("EVENT_ELAPSED_TIME: Elapsed time from process execution start to the event."),
                                        dbc.ListGroupItem("EVENT_FLOW_TIME:"),
                                        dbc.ListGroupItem("EVENT_ACTIVITY: Activity that is performed in the event."),
                                        dbc.ListGroupItem("EVENT_NUM_OF_OBJECTS: Number of objects involved in the event."),
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