# Import necessary libraries
from dash import html, dcc
import dash_bootstrap_components as dbc

# clustering evaluation explanation
clustering_evaluation_explanation = dbc.Card( 
                                            dbc.CardBody("Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.."),
                                            className="mb-3",
                                            )

# silhouette analysis explanation
silhouette_explanation = dbc.Card(
                                dbc.CardBody("Intuitively, the silhouette score quantifies the space between different clusters. For each number of clusters (up to the inserted max.), the selected clustering method is performed. Afterwards, it is measured how similar the observation are to the assigned cluster and how dissimilar they are to the observation of the nearest cluster. The plot displays the average silhouette score for each number of clusters. The measure has the range [-1,+1], whereas a score near +1 indicates that the clusters are well separated and negative scores indicate that the samples might be wrongly separated. Thus, to get a reasonable clustering result, one should choose the cluster number with the maximal positive average silhouette score."),
                                className="mb-3",
                                )

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