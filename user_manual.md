# ICCL

Introduction here

## Table of Content

1. [Installation](#installation) 
2. [Application Layout](#application-layout)
3. [Data Management](#data-management)
	* [Import OCEL](#import-ocel)
	* [View OCEL](#view-ocel)
4. [Configuration Pipeline](#configuration-pipeline)
	* [Feature Selection](#feature-selection)
	* [Graph Embedding](#graph-embedding)
	* [Clustering](#clustering)
	* [Automatic Clustering](#automatic-clustering)
	* [Cluster Evaluation](#cluster-evaluation)
	* [Process Executions](#process-executions)
5. [Process Discovery](#process-discovery)
	* [Discover Process Model](#discover-process-model)
	* [Conformance Checking](#conformance-checking)
	* [Export Process Model](#export-process-model)
6. [Algorithms](#algorithms)
	* [Process Execution Extraction](#process-execution-extraction)
	* [Graph Embedding Methods](#graph-embedding-methods)
	* [Clustering Techniques](#clustering-techniques)
	* [Evaluation Measures](#evaluation-measures)
		* [Cluster Evaluation](#cluster-evaluation)
		* [Model Evaluation](#model-evaluation)
	* [Model Discovery](#model-discovery)
	

## Installation

## Application Layout
* shortly describe our pages
* how can you access help page

## Data Management
### Import OCEL
![plot](./assets/import_data.png)

You have two options to import your OCEL:
* 1a: click on the drag and drop field and select a file from the file browser
* 1b.1: insert the path in which the OCEL is located
* 1b.2: click the SEARCH button
* 1b.3: select the OCEL from the list. Note: only files with .csv, .jsonocel or .xmlocel extensions are listed

If you have selected an OCEL with .csv format, the first rows of the .csv are displayed in table 6 and you have to fill out the form 2:
* 2a: specify the OCEL parameters by selecting the column names in the dropdown lists. Select the object types in the last dropdown list.
* 2b: parse the parameters by clicking the PARSE CSV PARAMETERS button

After selecting a file, proceed as follows: 
* 3: specify the type of process execution extraction. If leading object type is selected, the possible object types are loaded into the list. Select one of the object types.
* 4: click the UPLOAD button. Note: a progress bar will be displayed, showing you the progress of uploading the OCEL

Optional:
* 5: click the CANCEL button to stop the uploading process.

After successfully uploading the OCEL, the first five rows are displayed in the table 6.

Note: only OCEL files with .csv, .jsonocel or .xmlocel extensions are supported.
### View OCEL


## Configuration Pipeline

### Feature Selection

### Graph Embedding

### Clustering

### Automatic Clustering

### Cluster Evaluation

### Process Executions

## Process Discovery

### Discover Process Model

### Conformance Checking

### Export Process Model

## Algorithms

Here you can find a short description of the algorithms that are used in ICCL.

### Process Execution Extraction

A process execution is a set of events of connected objects and resembles the case notion in traditional Process Mining. You can choose between two different types of extraction methods:

*  **Connected Components**: the technique uses the object graphs and extracts a process execution based on all transitively connected objects. This way, complex event logs might lead to large process executions.

*  **Leading Object Type**: after choosing one object type as the leading type, a process execution is constructed by looking at each object of the object type. Objects that are connected to the leading object are added to the process execution unless another object of that type has a lower distance. As a result, the process executions are more limited in size.

### Graph Embedding Methods

*  **Custom Feature Graph Embedding**: especially designed for ICCL. The algorithm first creates a node embedding via FeatherNode which uses characteristic functions of node features with random walk weights to describe node neighborhoods. In the second step, the node embeddings are averaged over each dimension, resulting in a vectorized embedding of the graph. Focusses **features**, the structure of the process execution graphs is only implicitly considered. Check out the [karateclub documentation](https://karateclub.readthedocs.io/en/latest/_modules/karateclub/node_embedding/attributed/feathernode.html) for more information about the FeatherNode parameters that can be configured. 

*  **Graph2Vec**: first identifies subgraphs sourrounding each node in the feature graphs. By means of the Weisfeiler-Lehman’s algorithm, the subgraphs are considered as the vocabulary for a doc2vec SkipGram model. Since the graph’s structure is captured within the algorithm, feature graphs that are similar in structure will be close in the embedding space. Focusses the **graph structure** and additionally allows one feature per node. Check out the [karateclub documentation](https://karateclub.readthedocs.io/en/latest/_modules/karateclub/graph_embedding/graph2vec.html) for more information about the parameters that can be configured. 

*  **Feather-G**: uses characteristic functions defined on graph vertices to describe the distribution of vertex attributes. Feather-G extracts node-level features that are pooled and then used to create a description of the feature graph. Focusses only the **graph structure**. Check out the [karateclub documentation](https://karateclub.readthedocs.io/en/latest/_modules/karateclub/graph_embedding/feathergraph.html) for more information about the parameters that can be configured. 

### Clustering Techniques
ICCL makes use of the sklearn.cluster module to apply different clustering algorithms to the embedding. Please refer to the [sklearn.cluster documentation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster) to gather more information about the clustering parameters that can be configured in ICCL. 
*  **K-Means**: clusters data by trying to separate samples in k groups of equal variance. It aims to choose centroids that minimize the within-cluster sum-of-squares. The number of clusters (k) has to be specified beforehand.

*  **Hierarchical Clustering**: builds nested clusters by merging or splitting them successively. This hierarchy of clusters is represented as a tree, whereas the root is the unique cluster gathering all samples and the leaves are clusters containing only one sample. In ICCL, you have to specify the number of clusters beforehand.

*  **Mean-Shift**: aims to discover blobs in a smooth density of samples. It is a centroid based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region.

*  **Affinity-Propagation**: Affinity-Propagation creates clusters by sending messages between pairs of samples until convergence. The algorithm finds members of the data points that are representatives of the clusters.

*  **DBscan**: Density-Based Spatial Clustering of Applications with Noise is an clustering approach that finds core samples of high density and expands clusters from them. The parameter epsilon is specifying the radius of a neighborhood with respect to some point, in which the number of neighboring points is counted.

### Evaluation Measures

#### Cluster Evaluation

*  **Silhouette Score**: quantifies the space between different clusters. For each number of clusters, a clustering algorithm is performed. Afterwards, it is measured how similar the observation are to the assigned cluster and how dissimilar they are to the observation of the nearest cluster. The measure has the range [-1,+1], whereas a score near +1 indicates that the clusters are well separated and negative scores indicate that the samples might be wrongly separated. Generally, the silhouette score is calculated for each datapoint and then averaged over the whole dataspace. You can find the calculation steps here: [click](https://en.wikipedia.org/wiki/Silhouette_(clustering))

*  **Davies-Bouldin Index**: measure of the ratio between within-cluster distances, and between cluster distances. The score is bounded between [0, 1]. The lower the value, the tighter the clusters and the seperation between clusters. The steps of calculation can be found here: [click](https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index)

#### Model Evaluation

*  **Fitness**: measures to what extend the observed traces can be replayed by the model.

*  **Precision**: measures the extend of behaviour that is not captured in the event log, but allowed in the model.

### Model Discovery

The discovery of an object-centric Petri net works as follows: for each object type, a flattened event log is created and subsequently an accepting Petri net is discovered. For the discovery part, ICCL supports the Inductive Miner. In the second step, the Petri nets are merged into one Petri net. The object types are assigned to places and the variable arcs are identified.

**Inductive Miner**: the algorithm recursively performs the following steps
* build a directly follows graph from the log
* find a cut and return the cut-operator and cut-partition
* split the log into sublogs
