from sklearn.cluster import DBSCAN 
from sklearn.cluster import MeanShift, KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from ocpa.algo.util.filtering.log import case_filtering
from ocpa.objects.log.util import misc as log_util

import matplotlib.pyplot as plt
import io
import base64
import clusteval

def perform_silhouette_analysis(X, max_clusters, method):
    silhouette = []
    for i in range(2,max_clusters+1):
        if method == 'K-Means':
            kmeans = KMeans(n_clusters = i)
            kmeans.fit(X)
            labels = kmeans.labels_
            silhouette.append(silhouette_score(X, labels))
        elif method == 'Hierarchical':
            hierarchical_cluster = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='ward')
            labels = hierarchical_cluster.fit_predict(X)
            silhouette.append(silhouette_score(X, labels))

    return silhouette 

def perform_DBSCAN(X, parameters):
    labels = DBSCAN(eps=parameters['eps'], min_samples=parameters['min_samples']).fit_predict(X)

    return labels

def perform_AffinityPropagation(X, parameters):
    labels = AffinityPropagation(max_iter=parameters['max_iter'],convergence_iter=parameters['convergence_iter']).fit_predict(X)

    return labels


def perform_MeanShift(X, parameters):
    clustering = MeanShift(max_iter=parameters['max_iter']).fit(X)

    return clustering.labels_

def perform_KMeans(X, n_clusters, parameters):
    clustering = KMeans(n_clusters, n_init=parameters['n_init'], max_iter=parameters['max_iter']).fit(X)

    return clustering.labels_

def perform_HierarchicalClustering(X, n_clusters, parameters):
    hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=parameters['linkage'])
    labels = hierarchical_cluster.fit_predict(X)
    
    return labels

def create_clustered_df(process_executions, labels):
    clustered_df = pd.DataFrame(columns=['id', 'cluster'])
    num_process_executions = len(process_executions)
    clustered_df['id'] = [i for i in range(0,num_process_executions)]
    clustered_df['cluster'] = labels

    return clustered_df

def partition_ocel(ocel, ocel_df, clustered_df):
    sub_ocels = [] 
    
    for cluster in np.sort(clustered_df['cluster'].unique()):   #sort because .unique() otherwise sorts after first occurences
        event_ids = []
        cluster_df = clustered_df[clustered_df['cluster'] == cluster]
        cluster_process_ex = list(cluster_df.index.values)
        for i, process_ex in enumerate(ocel.process_executions):
            if i in cluster_process_ex:
                event_ids.append(list(process_ex))
        # flatten list
        event_ids = [item for sublist in event_ids for item in sublist]
        ocel_df_cluster = ocel_df[ocel_df['event_id'].isin(event_ids)]
        ocel.parameters = {"obj_names": ocel.object_types, "val_names":[], "act_name": "event_activity", "time_name":"event_timestamp"}
        # to-do: add parameters so that conversion is possible
        new_log = log_util.copy_log_from_df(ocel_df_cluster, ocel.parameters)
        sub_ocels.append(new_log)
    
    return sub_ocels

def get_cluster_summary(clustered_df):

    summary_df = pd.DataFrame(columns=['Cluster ID', 'Number of Process Executions'])
    summary_df['Cluster ID'] = np.sort(clustered_df['cluster'].unique())    #sort cluster df ascending
    summary_df['Number of Process Executions'] = clustered_df['cluster'].value_counts().sort_index()    #sort by index (by cluster) to match with summary_df
    #summary_df['Number of Process Executions'] = clustered_df['cluster'].map(clustered_df['cluster'].value_counts())   #returns the number of process executions in first cluster for all clusters in summary

    return summary_df


def cluster_evaluation_hierarchical(X, linkage):

    buf = io.BytesIO()
    plt.figure()
    fig, axs = plt.subplots(1,2, figsize=(10,4))

    if len(X) < 15:
        max_clust = len(X)
    else:
        max_clust = 15

    # dbindex
    results = clusteval.dbindex.fit(X, linkage=linkage, max_clust=max_clust)
    _ = clusteval.dbindex.plot(results, title='DBindex', ax=axs[0], visible=False)

    # silhouette
    results = clusteval.silhouette.fit(X, linkage=linkage, max_clust=max_clust)
    _ = clusteval.silhouette.plot(results, title='Silhouette', ax=axs[1], visible=False)

    plt.savefig(buf, format = "png")
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    buf.close()

    return "data:image/png;base64,{}".format(data)

def cluster_evaluation_kmeans(X):

    buf = io.BytesIO()

    plt.figure()
    fig, axs = plt.subplots(1,2, figsize=(10,4))

    if len(X) < 15:
        max_clust = len(X)
    else:
        max_clust = 15

    # dbindex
    results = clusteval.dbindex.fit(X, cluster='kmeans', max_clust=max_clust)
    _ = clusteval.dbindex.plot(results, title='DBindex', ax=axs[0], visible=False)

    # silhouette
    results = clusteval.silhouette.fit(X, cluster='kmeans', max_clust=max_clust)
    _ = clusteval.silhouette.plot(results, title='Silhouette', ax=axs[1], visible=False)

    plt.savefig(buf, format = "png")
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    buf.close()

    return "data:image/png;base64,{}".format(data)

def cluster_evaluation_dbscan(X):

    buf = io.BytesIO()

    plt.figure()
    fig, ax = plt.subplots(figsize=(7, 4))

    # dbscan
    results = clusteval.dbscan.fit(X)
    _ = clusteval.dbscan.plot(results, title='DBscan', ax=ax, visible=False)

    plt.savefig(buf, format = "png")
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    buf.close()

    return "data:image/png;base64,{}".format(data)