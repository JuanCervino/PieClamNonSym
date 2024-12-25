# PieClam
The code for the paper PieClam - Inclusive Exclusive Clustier Affiliation Model with Prior.
To run the experiments in the paper, use the designated jupyter notebooks in the experiments folder.

# Data structure
The graphs are represented as pytorch geometric data objects with added member variables depending on the experiment conducted. In all experiments data.x represents the affiliation features learned by the algorithm.
### Anomaly Detection
- Node attributes: data contains a data.attr member that is the node data features. data.attr is used only in the anomaly detection experiment, as explained in the paper.
- data.gt_anomalous is a list representing the ground truth anomalies. 

### Link Prediction
- For the reparametrization algorithm the omitted non edges are inserted into the edge_index array of the data object. The omitted members of the edge_index array (edges and non-edges) are denoted using the data.edge_attr member. Omitted dyads have edge_attr == 0 and retained edges have edge_attr == 1 (retained non edges are not added to the edge_index array).

# Hyper Parameter Configuration
The hyper-parameters are organized in the \hypers folder.
There is a separate hypers.yaml file for every task, and in each file there is a global configuration dictionary for each of the four Clam models which is relevant for all datasets, and a per-dataset optimal configuration for all of the models. The per-dataset configurations act as a delta for the global configuration, where if a parameter does not appear in the delta it is given by the global config. 
The hyperparameters are organized in four groups: clamiter_init (initializations for clamiter), feat_opt (feature optimization parameters), prior_opt (prior optimization parameters) and back_forth (alternation hyperparameters).
In addition, there is an option to set the hyperparameters manually before the optimization by adding "config_triplets" list of lists where each list is a triplet for which the first element is the outer group (clamiter_init, feat_opt, prior_opt or back_forth), the second element is the hyperparameter name (e.g. n_iter, lr, etc....) and the third element is the value. Please see examples in the \experiments directory.

# Datasets
Synthetic datasets are simulated in the simulations.py file. The real world datasets are not part of the github repository, but have a specific loading mechanism via the datasets/import_datasets.import_dataset function. The datasets can be obtained and loaded in the following way:
- Texas is part of the WebKB dataset that is available in pytorch geometric and is downloaded automatically with the import_dataset function.
- JH55 (John's Hopkins 55) is part of the facebook 100 dataset, and should be in .mat format.
- Squirrel is part of the wikipedia dataset and is available at snap https://snap.stanford.edu/data/wikipedia-article-networks.html.
- Reddit, Elliptic and Photo are available from the GGAD project that can be found here: https://github.com/mala-lab/GGAD/tree/main






