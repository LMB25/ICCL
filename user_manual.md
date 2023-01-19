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
Here you can find a short description of the algorithms that are supported by ICCL.
### Process Execution Extraction
A process execution is a set of events of connected objects and resembles the case notion in traditional Process Mining. You can choose between two different types of extraction methods:
* **Connected Components**: the technique uses the object graphs and extracts a process execution based on all transitively connected objects. This way, complex event logs might lead to large process executions.
* **Leading Object Type**: after choosing one object type as the leading type, a process execution is constructed by looking at each object of the object type. Objects that are connected to the leading object are added to the process execution unless another object of that type has a lower distance. As a result, the process executions are more limited in size.
### Graph Embedding Methods
* **Custom Feature Graph Embedding**: especially designed for ICCL. The algorithm first creates a node embedding via FeatherNode which uses characteristic functions of node features with random walk weights to describe node neighborhoods. In the second step, the node embeddings are averaged over each dimension, resulting in a vectorized embedding of the graph. Focusses **features**, the structure of the process execution graphs is only implicitly considered.
* **Graph2Vec**:  first identifies subgraphs sourrounding each node in the feature graphs. By means of the Weisfeiler-Lehman’s algorithm, the subgraphs are considered as the vocabulary for a doc2vec SkipGram model. Since the graph’s structure is captured within the algorithm, feature graphs that are similar in structure will be close in the embedding space. Focusses the **graph structure** and additionally allows one feature per node.
* **Feather-G**:  uses characteristic functions defined on graph vertices to describe the distribution of vertex attributes. Feather-G extracts node-level features that are pooled and then used to create a description of the feature graph. Focusses only the **graph structure**.
### Clustering Techniques
 * **K-Means**: clusters data by trying to separate samples in k groups of equal variance. It aims to choose centroids that minimize the within-cluster sum-of-squares. The number of clusters (k) has to be specified beforehand.
 * **Hierarchical Clustering**: builds nested clusters by merging or splitting them successively. This hierarchy of clusters is represented as a tree, whereas the root is the unique cluster gathering all samples and the leaves are clusters containing only one sample. In ICCL, you have to specify the number of clusters beforehand.
 * **Mean-Shift**: aims to discover blobs in a smooth density of samples. It is a centroid based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region.
 * **Affinity-Propagation**: Affinity-Propagation creates clusters by sending messages between pairs of samples until convergence. The algorithm finds members of the data points that are representatives of the clusters.
 * **DBscan**: Density-Based Spatial Clustering of Applications with Noise is an clustering approach that finds core samples of high density and expands clusters from them. The parameter epsilon is specifying the radius of a neighborhood with respect to some point, in which the number of neighboring points is counted.
### Evaluation Measures
#### Cluster Evaluation
* **Silhouette Score**:  quantifies the space between different clusters. For each number of clusters, a clustering algorithm is performed. Afterwards, it is measured how similar the observation are to the assigned cluster and how dissimilar they are to the observation of the nearest cluster. The measure has the range [-1,+1], whereas a score near +1 indicates that the clusters are well separated and negative scores indicate that the samples might be wrongly separated.
* **Davies-Bouldin Index**: measure of the ratio between within-cluster distances, and between cluster distances. The score is bounded between [0, 1]. The lower the value, the tighter the clusters and the seperation between clusters.")
#### Model Evaluation
* **Fitness**:
* **Precision**:
### Model Discovery
* **Inductive Miner**:

