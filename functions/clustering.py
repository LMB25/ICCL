from sklearn.cluster import DBSCAN 
from sklearn.cluster import MeanShift
import numpy as np
import pandas as pd
from ocpa.algo.util.filtering.log import case_filtering
from ocpa.objects.log.util import misc as log_util


def perform_DBSCAN(X):
    labels = DBSCAN(eps=0.75, min_samples=8).fit_predict(X)

    return labels


def perform_MeanShift(X):
    clustering = MeanShift().fit(X)

    return clustering.labels_

def create_clustered_df(process_executions, labels):
    clustered_df = pd.DataFrame(columns=['id', 'cluster'])
    num_process_executions = len(process_executions)
    clustered_df['id'] = [i for i in range(0,num_process_executions)]
    clustered_df['cluster'] = labels

    return clustered_df

def partition_ocel(ocel, clustered_df):
    sub_ocels = [] 

    for cluster in clustered_df['cluster'].unique():
        cluster_df = clustered_df[clustered_df['cluster'] == cluster]
        cluster_process_ex = list(cluster_df.index.values)
        # get corresponding event ids
        event_ids = [key for (key,val) in ocel.process_execution_mappings.items() if val in cluster_process_ex]
        new_event_df = ocel.log.log.loc[ocel.log.log["event_id"].isin(event_ids)].copy()
        ocel.parameters = {"obj_names": ocel.object_types, "val_names":[], "act_name": "event_activity", "time_name":"event_timestamp"}
        # to-do: add parameters so that conversion is possible
        new_log = log_util.copy_log_from_df(new_event_df, ocel.parameters)
        sub_ocels.append(new_log)
    
    return sub_ocels