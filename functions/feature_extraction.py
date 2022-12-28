from ocpa.algo.predictive_monitoring import factory as predictive_monitoring
import ocpa.algo.predictive_monitoring.event_based_features.extraction_functions as event_features
import ocpa.algo.predictive_monitoring.execution_based_features.extraction_functions as execution_features
from ocpa.algo.predictive_monitoring import tabular, sequential
import pandas as pd

# event_based_mapping = {
#                 "EVENT_ELAPSED_TIME": event_features.elapsed_time,
#                 "EVENT_REMAINING_TIME": event_features.remaining_time,
#                 "EVENT_FLOW_TIME": event_features.flow_time
#               }

# execution_based_mapping = {
#                     "EXECUTION_NUM_OF_EVENTS": execution_features.number_of_events,
#                     "EXECUTION_NUM_OF_END_EVENTS":execution_features.number_of_ending_events,
#                     "EXECUTION_THROUGHPUT":execution_features.throughput_time,
#                     "EXECUTION_IDENTITY":execution_features.execution,
#                     "EXECUTION_NUM_OBJECT":execution_features.number_of_objects,
#                     "EXECUTION_UNIQUE_ACTIVITIES":execution_features.unique_activites,
#                     "EXECUTION_NUM_OF_STARTING_EVENTS":execution_features.number_of_starting_events,
#                     "EXECUTION_LAST_EVENT_TIME_BEFORE":execution_features.delta_last_event,
#                 }


def create_feature_set(ocel, event_feature_list, extraction_feature_list):
    feature_set_event = []
    if "EVENT_ACTIVITY" in event_feature_list:
        activities = list(set(ocel.log.log["event_activity"].tolist()))
        feature_set_event += [(predictive_monitoring.EVENT_ACTIVITY, (act,)) for act in activities]
        event_feature_list.remove("EVENT_ACTIVITY")
    if event_feature_list != []:
        event_feature_list = [getattr(predictive_monitoring,key) for key in event_feature_list]
        feature_set_event += [(event_feature, ()) for event_feature in event_feature_list]
        
    if extraction_feature_list != []:
        extraction_feature_list = [getattr(predictive_monitoring,key) for key in extraction_feature_list]
        feature_set_extraction = [(object_feature, ()) for object_feature in extraction_feature_list]
    else:
        feature_set_extraction = []
    
    return feature_set_event, feature_set_extraction

def extract_features(ocel_log, feature_set_event, feature_set_extr, repr):
    feature_set_event, feature_set_extr = create_feature_set(ocel_log, feature_set_event, feature_set_extr)
    feature_storage = predictive_monitoring.apply(ocel_log, feature_set_event, feature_set_extr)
    if repr == 'graph':
        extraction = feature_storage
    elif repr == 'table':
        extraction = tabular.construct_table(feature_storage)
    elif repr == 'sequential':
        extraction = sequential.construct_sequence(feature_storage)
    return extraction


def create_extraction_feature_dfs(ocel):
    feature_options_extraction = ['EXECUTION_NUM_OF_EVENTS', 'EXECUTION_NUM_OF_END_EVENTS', 'EXECUTION_THROUGHPUT', 'EXECUTION_NUM_OBJECT', 'EXECUTION_UNIQUE_ACTIVITIES', 'EXECUTION_NUM_OF_STARTING_EVENTS', 'EXECUTION_LAST_EVENT_TIME_BEFORE']
    feature_options_extraction_renamed = ["Number of Events", "Number of Ending Events", "Throughput Duration", "Number of Objects", "Unique Activities", "Number of Starting Events", "Duration of Last Event"]
    extraction_feature_list = [getattr(predictive_monitoring,key) for key in feature_options_extraction]
    feature_set_extraction = [(object_feature, ()) for object_feature in extraction_feature_list]
    feature_storage = predictive_monitoring.apply(ocel, [], feature_set_extraction)
    extraction_value_list = []
    for graph in feature_storage.feature_graphs:
        values = [] 
        for tuple_feature in feature_set_extraction:
            values.append(graph.attributes[tuple_feature])
        extraction_value_list.append(values)
    return extraction_value_list