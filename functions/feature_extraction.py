from ocpa.algo.predictive_monitoring import factory as predictive_monitoring
import ocpa.algo.predictive_monitoring.event_based_features.extraction_functions as event_features
import ocpa.algo.predictive_monitoring.execution_based_features.extraction_functions as execution_features
from ocpa.algo.predictive_monitoring import tabular, sequential


event_based_mapping = {
                "EVENT_ELAPSED_TIME": event_features.elapsed_time,
                "EVENT_REMAINING_TIME": event_features.remaining_time,
                "EVENT_FLOW_TIME": event_features.flow_time
              }

execution_based_mapping = {
                    "EXECUTION_NUM_OF_EVENTS": execution_features.number_of_events,
                    "EXECUTION_NUM_OF_END_EVENTS":execution_features.number_of_ending_events,
                    "EXECUTION_THROUGHPUT":execution_features.throughput_time,
                    "EXECUTION_IDENTITY":execution_features.execution,
                    "EXECUTION_NUM_OBJECT":execution_features.number_of_objects,
                    "EXECUTION_UNIQUE_ACTIVITIES":execution_features.unique_activites,
                    "EXECUTION_NUM_OF_STARTING_EVENTS":execution_features.number_of_starting_events,
                    "EXECUTION_LAST_EVENT_TIME_BEFORE":execution_features.delta_last_event,

                }


def create_feature_set(event_feature_list, extraction_feature_list):
    if event_feature_list != []:
        event_feature_list = [event_based_mapping[key] for key in event_feature_list]
        feature_set_event = [(event_feature, ()) for event_feature in event_feature_list]
    else:
        feature_set_event = []
    if extraction_feature_list != []:
        extraction_feature_list = [execution_based_mapping[key] for key in extraction_feature_list]
        feature_set_extraction = [(object_feature, ()) for object_feature in extraction_feature_list]
    else:
        feature_set_extraction = []
    
    return feature_set_event, feature_set_extraction

def extract_features(ocel_log, feature_set_event, feature_set_extr, repr):
    feature_storage = predictive_monitoring.apply(ocel_log, feature_set_event, feature_set_extr)
    if repr == 'graph':
        extraction = feature_storage
    elif repr == 'table':
        extraction = tabular.construct_table(feature_storage)
    elif repr == 'sequential':
        extraction = sequential.construct_sequence(feature_storage)
    return extraction
