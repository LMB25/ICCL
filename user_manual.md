# ICCL

Welcome! This manual contains instructions how to use the ICCL tool. The aim of this project was to provide a process discovery tool that enhances the comprehensibility of discovered object-centric Petri nets by clustering.
The prerequisites to apply this tool are knowledge in Process Mining, object-centric Process Mining and unsupervised learning techniques such as clustering.
The main pipeline of the application works the following way:
1. After the installation and successfull start of the server, the user can open the application. By default, the user gets on the data import page.
2. On the import page the user can upload an object-centric event log or a csv file from their local machine. The user can load several files and can select from the directory.
3. The user then sets the configuration. This includes configuring the set of selected features, the graph embedding method, the clustering method and number of clusters as well as cluster evaluation.
4. The user applies the process discovery algorithms and receives several process models based on the clustering before. The user now have the possibility to have a more comprehensible representation of the original process model.


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
### Using Docker
The code comes with a Dockerfile to build and run the application inside a docker container. To build the container run
```
docker build -t docker-iccl . 
```
After the container is build the webapp can be run using
```
docker run -p 8050:8050 docker-iccl
```


## Application Layout
![Application_Layout_Default_Start.png](imgs%2FApplication_Layout_Default_Start.png)
By default, the user starts his ICCL journey with the data import page. 
To navigate through the application pipeline, the user can use the sidebar under the application logo on the left.
The sidebar is visible throughout the whole workflow. The user can navigate back-and-forth using these tabs.
On the navigation bar with the RWTH Aachen logo we have on the right the help bar which provides the manual for this tool.


## Data Management
### Import OCEL
![Application_Layout_Data_import.png](imgs%2FApplication_Layout_Data_import.png)
The data import page is set by default but also reachable over the sidebars 'Import Data' button.
In the upper section of the page the user can feed in the data from its local machine via 'Drag & Drop' or by inserting a path of the file.
If selected the user can select the files he wants to use.
In the middle section of the page, the user can choose the type of the process execution extraction.
An upload functionality is provided underneath. 
In the lower section of the page, the user can view his uploaded event log.


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
Hier könnten wir jeweils die Algorithmen, die wir in unserem Tool benutzen mit einer kurzen Erklärung auflisten.
### Process Execution Extraction

### Graph Embedding Methods

### Clustering Techniques
 * K-Means: abcdefg
 * 
### Evaluation Measures
#### Cluster Evaluation
* Silhouette Score:
* Davies-Bouldin Index:
#### Model Evaluation
* Fitness:
* Precision:
### Model Discovery
* Inductive Miner:

