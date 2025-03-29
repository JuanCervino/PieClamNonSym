
import torch
from torch_geometric.datasets import KarateClub, Actor, IMDB, Amazon
from torch_geometric.data import Data, HeteroData
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import SNAPDataset, WebKB
from torch_geometric.utils import to_networkx, to_dense_adj, to_undirected, remove_self_loops, is_undirected, contains_self_loops, remove_isolated_nodes, contains_isolated_nodes, subgraph
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import MaxAbsScaler

import json

import os
import scipy.io as sio
import scipy.sparse as sp
import numpy as np
import pandas as pd
import random
from collections import Counter

from ogb.linkproppred import PygLinkPropPredDataset

from datasets.simulations import simulate_dataset
from datasets.data_utils import intersecting_tensor_from_non_intersecting_vec
from utils.printing_utils import printd
from utils import utils as my_utils



def import_dataset(dataset_name, remove_data_feats=True, verbose=False):
    '''will import a dataset with the same name as the dataset_name parameter'''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if dataset_name == 'ogbl-ddi':
        '''this dataaset is only for link prediction.
        the dataset comes with an edge index and a split with a train edge index set that is the directed version of the edge index.
        the dataset doesn't contain isolated nodes or any duplicate edges.'''

        dataset = PygLinkPropPredDataset(name = dataset_name, root = '../datasets/OGB') 
        split_edge = dataset.get_edge_split()
        valid_edge, test_edge = split_edge["valid"], split_edge["test"]
        
        data = dataset[0]
        #! problem: to_undirected changes the order of the edge_index so it's not easy to decide what to do with a given val and test set. maybe keep the initial list and pass that to the detector?

        data.val_edge_gt = valid_edge #debug
        data.test_edge_gt = test_edge #debug

        # this method of to_undirected doesn't rearange the edge index
        data.val_dyads_to_omit = (my_utils.undirect_unsorted(valid_edge['edge'].t()), 
                             my_utils.undirect_unsorted(valid_edge['edge_neg'].t()))
        data.test_dyads_to_omit = (my_utils.undirect_unsorted(test_edge['edge'].t()), 
                              my_utils.undirect_unsorted(test_edge['edge_neg'].t()))
        
        new_edge_index = torch.cat([data.edge_index, 
                                    data.val_dyads_to_omit[0], 
                                    data.test_dyads_to_omit[0]], 
                                   dim=1)
        
        data.edge_attr = torch.ones(new_edge_index.shape[1])
        data.edge_index = new_edge_index

    elif dataset_name == 'squirrel':
        edges_df = pd.read_csv('../datasets/wikipedia/squirrel/musae_squirrel_edges.csv')
        edges = edges_df.values.T  # Transpose to shape [2, num_edges]
        edge_index = torch.tensor(edges, dtype=torch.long)

        # Load labels
        labels_df = pd.read_csv('../datasets/wikipedia/squirrel/musae_squirrel_target.csv')
        labels = labels_df['target'].values  # Assuming the label column is named 'target'
        y = torch.tensor(labels, dtype=torch.long)

        # Load sparse features
        with open('../datasets/wikipedia/squirrel/musae_squirrel_features.json') as f:
            features_dict = json.load(f)

        # Assuming there are N nodes and each node's features are stored as a list of non-zero indices.
        num_nodes = len(features_dict)
        num_features = max(max(indices) for indices in features_dict.values()) + 1  # Assuming feature indices start at 0


        # Create the PyTorch Geometric Data object
        edge_index = remove_self_loops(edge_index)[0]
        edge_index = remove_isolated_nodes(edge_index)[0]
        edge_index = to_undirected(edge_index)
        
        data = Data(x=None, edge_index=edge_index, y=y)

    

    elif dataset_name == 'crocodile':
        edges_df = pd.read_csv('../datasets/wikipedia/crocodile/musae_crocodile_edges.csv')
        edges = edges_df.values.T  # Transpose to shape [2, num_edges]
        edge_index = torch.tensor(edges, dtype=torch.long)

        # Load labels
        labels_df = pd.read_csv('../datasets/wikipedia/crocodile/musae_crocodile_target.csv')
        labels = labels_df['target'].values  # Assuming the label column is named 'target'
        y = torch.tensor(labels, dtype=torch.long)

        # Load sparse features
        with open('../datasets/wikipedia/crocodile/musae_crocodile_features.json') as f:
            features_dict = json.load(f)

        # Assuming there are N nodes and each node's features are stored as a list of non-zero indices.
        num_nodes = len(features_dict)
        num_features = max(max(indices) for indices in features_dict.values()) + 1  # Assuming feature indices start at 0
        raw_attr = sp.lil_matrix((num_nodes, num_features))

        for node_id, features in features_dict.items():
            raw_attr[node_id, :] = features

        edge_index = remove_self_loops(edge_index)[0]
        edge_index = remove_isolated_nodes(edge_index)[0]
        edge_index = to_undirected(edge_index)

        # Create the PyTorch Geometric Data object
        data = Data(x=None, raw_attr=raw_attr, edge_index=to_undirected(edge_index), y=y)

    

    elif dataset_name == 'texas':
        data = WebKB(root=os.path.join(current_dir,'WebKB'), name='texas')[0]
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = remove_isolated_nodes(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)
        dense_attr_np = data.x.numpy()
        data.raw_attr = sp.lil_matrix(dense_attr_np)
        
        if hasattr(data, 'y'):
            data.y = intersecting_tensor_from_non_intersecting_vec(data.y)

        data.x = None

    elif dataset_name == 'cornell':
        data = WebKB(root=os.path.join(current_dir,'WebKB'), name='cornell')[0]
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = remove_isolated_nodes(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)
        data
        if hasattr(data, 'y'):
            data.y = intersecting_tensor_from_non_intersecting_vec(data.y)

        data.x = None

    elif dataset_name == 'wisconsin':
        data = WebKB(root=os.path.join(current_dir,'WebKB'), name='wisconsin')[0]
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = remove_isolated_nodes(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)
        if hasattr(data, 'y'):
            data.y = intersecting_tensor_from_non_intersecting_vec(data.y)

        data.x = None

    elif dataset_name == 'BlogCatalog':
        data = load_data_matlab_format('BlogCatalog', 'BlogCatalog')
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)
    
    elif dataset_name == 'ACM':
        data = load_data_matlab_format('ACM', 'ACM')
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)

    elif dataset_name == 'Flickr':
        data = load_data_matlab_format('Flickr', 'Flickr')
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)
     

    elif dataset_name == 'photo':
        data = load_data_matlab_format('Photo', 'photo')
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)

    elif dataset_name == 'reddit':
        data = load_data_matlab_format('Reddit','reddit')
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)

    elif dataset_name == 'elliptic':
        data = load_data_matlab_format('Elliptic','elliptic')
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)
   
    elif dataset_name == 'JohnsHopkins55':
        '''for .mat files'''
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dict = sio.loadmat(os.path.join(current_dir, f"{'facebook100'}/{dataset_name}.mat"))
        adj = sp.coo_matrix(data_dict['A'])
        edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
        raw_attr = sp.lil_matrix(data_dict['local_info'])

        data = Data(edge_index=edge_index, raw_attr=raw_attr, x=None)
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = remove_isolated_nodes(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)

   
   
    elif dataset_name == 'sbm3x3':
        data = simulate_dataset('sbm3x3', verbose=verbose)
        
    elif dataset_name == 'bipartite':
        data = simulate_dataset('bipartite', verbose=verbose)

    elif dataset_name == 'sbm3x3HalfCenter':
        data = simulate_dataset('sbm3x3HalfCenter', verbose=verbose)
        
    
    elif dataset_name == 'sbm3x3HalfDiag':
        data = simulate_dataset('sbm3x3HalfDiag', verbose=verbose)

    elif dataset_name == 'bipartiteHalf':
        data = simulate_dataset('bipartiteHalf', verbose=verbose)

    elif dataset_name == 'smallBipart':
        data = simulate_dataset('smallBipart', verbose=verbose)

    else:
        raise NotImplementedError(f'dataset {dataset_name} not implemented yet')
    

    data.edge_attr = torch.ones(data.edge_index.shape[1], dtype=torch.bool) 

    data.edge_index, data.edge_attr, non_isolated_mask = remove_isolated_nodes(data.edge_index, data.edge_attr)

    if hasattr(data, 'y'):
        if data.y is not None:
            data.y = data.y[non_isolated_mask]
    if hasattr(data, 'gt_nomalous'):
        #? tested?
        if data.gt_nomalous is not None:
            data.gt_nomalous = data.gt_nomalous[non_isolated_mask]
    
    if hasattr(data, 'raw_attr'):
        if data.raw_attr is not None:
            data.raw_attr = data.raw_attr[non_isolated_mask]
    
    data.num_nodes = data.edge_index.max().item() + 1
    if not hasattr(data, 'gt_nomalous'):
        data.gt_nomalous = torch.zeros(data.num_nodes).bool()
    data.name=dataset_name
    # 1 flag is normal: normal edge and normal node.

    assert not data.has_isolated_nodes(), 'in import_dataset: isolated nodes found'
    assert data.is_undirected(), 'in import_dataset: graph is not undirected'
    assert not data.has_self_loops(), 'in import_dataset: self loops found'

    return data


def load_data_(dataset, train_rate=0.3, val_rate=0.1):
    return load_data_matlab_format('anomaly_detection/GGAD_datasets', dataset, train_rate, val_rate)


def load_data_matlab_format(path_in_datasets, dataset, train_rate=0.3, val_rate=0.1):
    """loads a dataset in .mat format"""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data = sio.loadmat(os.path.join(current_dir, f"{path_in_datasets}/{dataset}.mat"))
    # dan: decompose data into labels attributes and network
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    # dan: these are two sparse matrix representations each with it's advantages.
    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    ano_labels = np.squeeze(np.array(label))
    
    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    # DAN: select train val test randomly
    random.shuffle(all_idx)
    train_idx = all_idx[: num_train]
    val_idx = all_idx[num_train: num_train + num_val]
    test_idx = all_idx[num_train + num_val:]

    # Sample some labeled normal nodes
    # *DAN: all_normal_label_idx is all normals in TRAIN only
    train_normal_idx = [i for i in train_idx if ano_labels[i] == 0]
    rate = 0.5  # change train_rate to 0.3 0.5 0.6  0.8
    # *DAN: in practice, we take rate*train_rate normals for the extra special suff
    # normal label idx are the training set of normal nodes
    train_rate_normal_idx = train_normal_idx[: int(len(train_normal_idx) * rate)]

    # DAN: take half of the normals in the train set and then take another 5% of that to mimic anomalies.
    train_idx = torch.tensor(train_idx, dtype=torch.long).sort()[0]
    train_normal_idx = torch.tensor(train_rate_normal_idx, dtype=torch.long).sort()[0]
    val_idx = torch.tensor(val_idx, dtype=torch.long).sort()[0]
    test_idx = torch.tensor(test_idx, dtype=torch.long).sort()[0]
    # Convert the list of numpy.ndarrays to a single numpy.ndarray
    edge_index_np = np.array(adj.nonzero())

# Convert the numpy.ndarray to a tensor
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)

    # edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    x = torch.tensor(feat.toarray(), dtype=torch.float)
    # train node mask is the 
    train_node_mask = torch.zeros(num_node, dtype=torch.bool)
    train_node_mask[train_rate_normal_idx] = True

    # TEST
    test_normal_idx = [i for i in test_idx if ano_labels[i] == 0]
    test_anomalies_idx = [i for i in test_idx if ano_labels[i] == 1]

    test_normal_mask = torch.zeros(num_node, dtype=torch.bool)
    test_anomalies_mask = torch.zeros(num_node, dtype=torch.bool)
    
    test_normal_mask[test_normal_idx] = True
    test_anomalies_mask[test_anomalies_idx] = True

    #! there is a thing they did in  which was to take the train index and the train normal index as diffree
    raw_attr = feat
    
    data = Data(edge_index=edge_index, 
                train_normal_idx=train_normal_idx, 
                train_idx=train_rate_normal_idx, 
                test_idx=test_idx, 
                val_idx=val_idx, 
                train_node_mask=train_node_mask, 
                test_normal_mask=test_normal_mask, 
                test_anomalies_mask=test_anomalies_mask, 
                gt_nomalous=torch.from_numpy(~ano_labels.astype(bool)), 
                raw_attr=raw_attr)


    return data



def transform_attributes(attr, transform='auto', n_components=32, normalize=True):
    scaler = MaxAbsScaler()
    if normalize:
        attr = scaler.fit_transform(attr)
    
    density_ratio = attr.nnz/(attr.shape[0]*attr.shape[1]) 
    
    if transform == 'auto':
        if density_ratio < 0.4:
            transform = 'truncated_svd'
        else:
            transform = 'pca'

    n_components = min(n_components, attr.shape[1])
    
    if transform == 'truncated_svd':
        svd = TruncatedSVD(n_components=n_components)
        attr = svd.fit_transform(attr)

    elif transform == 'pca':
        attr = attr.toarray()
        pca = PCA(n_components=n_components)
        attr = pca.fit_transform(attr)
    
    elif transform == 'none':
        pass
        
    else:
        raise ValueError(f'unknown transform {transform}')
    
    

    return torch.from_numpy(attr).float()