from ocpa.algo.predictive_monitoring import factory as predictive_monitoring
import ocpa.algo.predictive_monitoring.event_based_features.extraction_functions as event_features
import ocpa.algo.predictive_monitoring.execution_based_features.extraction_functions as execution_features
from ocpa.algo.predictive_monitoring import tabular, sequential
import pandas as pd
import numpy as np


def create_event_feature_set(ocel, event_feature_list):
    feature_set_event = []
    activities = list(set(ocel.log.log["event_activity"].tolist()))
    objects = ocel.object_types
    if "EVENT_ACTIVITY" in event_feature_list:
        feature_set_event += [(predictive_monitoring.EVENT_ACTIVITY, (act,)) for act in activities]
        event_feature_list.remove("EVENT_ACTIVITY")
    if "EVENT_CURRENT_ACTIVITIES" in event_feature_list:
        feature_set_event += [(predictive_monitoring.EVENT_CURRENT_ACTIVITIES, (act,)) for act in activities]
        event_feature_list.remove("EVENT_CURRENT_ACTIVITIES") 
    if "EVENT_PREVIOUS_ACTIVITY_COUNT" in event_feature_list:
        feature_set_event += [(predictive_monitoring.EVENT_PREVIOUS_ACTIVITY_COUNT, (act,)) for act in activities]
        event_feature_list.remove("EVENT_PREVIOUS_ACTIVITY_COUNT") 
    if "EVENT_PRECEDING_ACTIVITES" in event_feature_list:
        feature_set_event += [(predictive_monitoring.EVENT_PRECEDING_ACTIVITES, (act,)) for act in activities]
        event_feature_list.remove("EVENT_PRECEDING_ACTIVITES") 
    if "EVENT_DURATION" in event_feature_list:
        if "event_start_timestamp" in ocel.log.log.columns:
            if ocel.log.log["event_start_timestamp"].dtypes != np.object:
                feature_set_event += [(predictive_monitoring.EVENT_DURATION, ("event_start_timestamp",))]
        event_feature_list.remove("EVENT_DURATION") 
    if "EVENT_WAITING_TIME" in event_feature_list:
        if "event_start_timestamp" in ocel.log.log.columns:
            if ocel.log.log["event_start_timestamp"].dtypes != np.object:
                feature_set_event += [(predictive_monitoring.EVENT_WAITING_TIME, ("event_start_timestamp",))]
        event_feature_list.remove("EVENT_WAITING_TIME") 
    if "EVENT_PREVIOUS_TYPE_COUNT" in event_feature_list:
        feature_set_event += [(predictive_monitoring.EVENT_PREVIOUS_TYPE_COUNT, (obj,)) for obj in objects]
        event_feature_list.remove("EVENT_PREVIOUS_TYPE_COUNT") 

    if event_feature_list != []:
        event_feature_list = [getattr(predictive_monitoring,key) for key in event_feature_list]
        feature_set_event += [(event_feature, ()) for event_feature in event_feature_list]
    
    return feature_set_event

def extract_features(ocel_log, feature_set_event, repr):
    feature_set_event = create_event_feature_set(ocel_log, feature_set_event)
    feature_storage = predictive_monitoring.apply(ocel_log, feature_set_event, [])
    if repr == 'graph':
        extraction = feature_storage
    elif repr == 'table':
        extraction = tabular.construct_table(feature_storage)
    elif repr == 'sequential':
        extraction = sequential.construct_sequence(feature_storage)
    return extraction


def create_extraction_features(ocel):
    feature_options_extraction = ['EXECUTION_NUM_OF_EVENTS', 'EXECUTION_NUM_OF_END_EVENTS', 'EXECUTION_THROUGHPUT', 'EXECUTION_NUM_OBJECT', 'EXECUTION_UNIQUE_ACTIVITIES', 'EXECUTION_NUM_OF_STARTING_EVENTS', 'EXECUTION_LAST_EVENT_TIME_BEFORE']
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

def create_cluster_feature_summary(ocels):
    average_feature_values = []
    for ocel in ocels: 
        extraction_values = create_extraction_features(ocel)
        average_values = [np.round(float(sum(feature))/len(feature), 2) for feature in zip(*extraction_values)]
        average_feature_values.append(average_values)
        print(average_values)
    return average_feature_values