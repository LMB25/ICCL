from sklearn.cluster import DBSCAN 
from sklearn.cluster import MeanShift, KMeans, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from ocpa.objects.log.util import misc as log_util
from ocpa.algo.util.filtering.log import case_filtering
from sklearn.exceptions import ConvergenceWarning
import warnings

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

def perform_auto_clustering(X):
    best_score = None
    for method in [perform_auto_DBSCAN, perform_auto_KMeans, perform_auto_MeanShift]:#, perform_KMeans, perform_DBSCAN, perform_HierarchicalClustering]:
        labels, params = method(X)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X, labels)
        # case that only one cluster is discovered
        else:
            score = 0
        if best_score is None or score>best_score:
            best_score = score
            best_labels = labels
            best_params = params
    
    return best_labels, best_params

def perform_DBSCAN(X, parameters):
    labels = DBSCAN(eps=parameters['eps'], min_samples=parameters['min_samples']).fit_predict(X)

    return labels

# we first try to find the optimal value for eps and with this value we try to optimize min_samples
def perform_auto_DBSCAN(X):
    best_score = None
    best_eps = None
    
    eps_range = [0.1, 0.2, 0.3, 0.4, 0.5]
    for eps in eps_range:
        clustering_params_dict = {"eps": eps, "min_samples": 5}
        labels = perform_DBSCAN(X, clustering_params_dict)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X, labels)
        # case that only one cluster is discovered
        else:
            score = 0
        #print(f"score for DBScan with eps={eps}: {score}")
        if best_score is None or score>best_score:
            best_score = score
            best_eps = eps
    
    best_score = None
    for min_samples in range(1,11):
        clustering_params_dict = {"eps": best_eps, "min_samples": min_samples}
        labels = perform_DBSCAN(X, clustering_params_dict)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X, labels)
        # case that only one cluster is discovered
        else:
            score = 0
        #print(f"score for DBScan with eps={best_eps} & min_samples={min_samples}: {score}")
        if best_score is None or score>best_score:
            best_score = score
            best_labels = labels
            best_min_samples = min_samples
    
    return best_labels, {"method":"DBSCAN", "eps": best_eps, "min_samples":min_samples}
        
        
        

def perform_AffinityPropagation(X, parameters):
    labels = AffinityPropagation(max_iter=parameters['max_iter'],convergence_iter=parameters['convergence_iter']).fit_predict(X)

    return labels


def perform_MeanShift(X, parameters):
    clustering = MeanShift(max_iter=parameters['max_iter']).fit(X)

    return clustering.labels_

def perform_auto_MeanShift(X):
    best_score = None
    for max_iter in range(100,500,50): 
        clustering_params_dict = {"max_iter":int(max_iter)} #default max_iter=300
        labels = perform_MeanShift(X, clustering_params_dict)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X, labels)
        # case that only one cluster is discovered
        else:
            score = 0
        #print(f"score for Mean Shift with max_iter={max_iter}: {score}")
        if best_score is None or score>best_score:
            best_score = score
            best_labels = labels
            best_params = {"method":"MeanShift", "max_iter":max_iter}
                
    return best_labels, best_params

def perform_KMeans(X, n_clusters, parameters={"n_init":int(10), "max_iter":300}):
    clustering = KMeans(n_clusters, n_init=parameters['n_init'], max_iter=parameters['max_iter']).fit(X)
    return clustering.labels_

# currently we only explore the parameter n_clusters
# n_init and max_iter had no effect on clustering results if chosen large enough (default values normally suffice)
def perform_auto_KMeans(X):
    best_score = None
    max_clusters = min(X.shape[0],20)           #we cannot use more clusters than we have process execution graphs to cluster
    
    for n_clusters in range(2,max_clusters):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                labels = perform_KMeans(X, n_clusters)
            # in the case of Convergence Warning (KMeans does not find more clusters, e.g. because all datapoints within clusters are equal), we stop exploring larger cluster sizes
            except ConvergenceWarning:
                break
        score = silhouette_score(X, labels)
        #print(f"score for KMeans with n_clusters={n_clusters}: {score}")
        if best_score is None or score>best_score:
            best_score = score
            best_labels = labels
            best_params = {"method": "KMeans", "n_clusters":n_clusters}
        
    return best_labels, best_params

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

def partition_ocel(ocel, clustered_df):
    sub_ocels = [] 
    
    for cluster in np.sort(clustered_df['cluster'].unique()):   #sort because .unique() otherwise sorts after first occurences
        event_ids = []
        cluster_df = clustered_df[clustered_df['cluster'] == cluster]
        cluster_process_ex = list(cluster_df.index.values)
        for i, process_ex in enumerate(ocel.process_executions):
            if i in cluster_process_ex:
                event_ids.append(list(process_ex))
        new_log = case_filtering.filter_process_executions(ocel, event_ids)
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
    _ = clusteval.dbindex.plot(results, title='Davies-Bouldin index', ax=axs[0], visible=False)

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
    _ = clusteval.dbindex.plot(results, title='Davies-Bouldin index', ax=axs[0], visible=False)

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