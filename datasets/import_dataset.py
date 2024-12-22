
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

from datasets.simulations import simulate_dataset
from datasets.data_utils import intersecting_tensor_from_non_intersecting_vec
from utils.printing_utils import printd
from utils import utils



def import_dataset(dataset_name, remove_data_feats=True, verbose=False):
    '''will import a dataset with the same name as the dataset_name parameter'''
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if dataset_name == 'squirrel':
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
    

    #todo: facebook 100
    #todo: twich gamers
    #todo: texas

    elif dataset_name == 'cora':
        cora = Data.from_dict(torch.load('../datasets/Cora/processed/data.pt')[0])
        cora.raw_attr = sp.lil_matrix(cora.x.numpy())
        cora.x = None
        cora.y = intersecting_tensor_from_non_intersecting_vec(cora.y)
        
        data = cora

    elif dataset_name == 'citeseer':
        citeseer = Data.from_dict(torch.load('../datasets/CiteSeer/processed/data.pt')[0])
        citeseer.raw_attr = citeseer.x
        citeseer.x = None
        data = citeseer
        if hasattr(data, 'y'):
            data.y = intersecting_tensor_from_non_intersecting_vec(data.y)


    elif dataset_name == 'pubmed':
        pubmed = Data.from_dict(torch.load('../datasets/PubMed/processed/data.pt')[0])
        pubmed.x = None
        data = pubmed
        if hasattr(data, 'y'):
            data.y = intersecting_tensor_from_non_intersecting_vec(data.y)

    elif dataset_name == 'actor':
        actor_ds = Actor(root=os.path.join(current_dir,'Actor'))[0]
        actor_ds.edge_index = remove_self_loops(actor_ds.edge_index)[0]
        actor_ds.edge_index = to_undirected(actor_ds.edge_index)
        #* actor dataset has non intersecting communities
        actor_ds.y = intersecting_tensor_from_non_intersecting_vec(actor_ds.y)
        actor_ds.x = None
        data = actor_ds
    
    
    elif dataset_name == 'facebook348':
        ''' in the facebook ego networks, the last node is the ego node.
        (circles, circle_batch) are (node,community) pairs correspondantly.
        We remone the ego node, nodes that don't have communities and isolated nodes.
        algo:
        1. find comunityless nodes
        2. find ego node
        3. union of the 2 and use pyg.subgraph
        4. remove isolated nodes'''
        #* density: 0.045
        ego_facebook_ds = SNAPDataset(root=os.path.join(current_dir,'FacebookEgo'), name='ego-facebook')
        data = ego_facebook_ds[2]
        
        num_nodes_tot = data.edge_index.max().item() + 1


            #*  v   FIND COMMUNITYLESS NODES  v        
        
        affiliation_pairs = torch.cat([data.circle.unsqueeze(1), data.circle_batch.unsqueeze(1)], dim=1)
        _, sorted_indices = torch.sort(affiliation_pairs[:,0])
        sorted_affiliation_pairs = affiliation_pairs[sorted_indices]

        num_circles = data.circle_batch.max().item() + 1
        
        # make inclusive y matrix
        node_circle_matrix = torch.zeros((num_nodes_tot, num_circles), dtype=torch.int32)

        # Make (circle, batch) into binary matrix 
        node_circle_matrix[data.circle, data.circle_batch] = 1
        not_communityless_nodes_mask = node_circle_matrix.sum(dim=1) != 0
        not_communityless_nodes_indices = torch.where(not_communityless_nodes_mask)[0]

                #*  v   FIND EGO  v
        # ego node is the largest node index
        ego_node_index = num_nodes_tot - 1
                
                #* v REMOVE COMMUNITYLESS AND EGO
        nodes_to_keep_indices = list(set(not_communityless_nodes_indices.tolist()) - {ego_node_index})
        nodes_to_keep_mask = torch.zeros(num_nodes_tot, dtype=torch.bool)
        nodes_to_keep_mask[list(nodes_to_keep_indices)] = True
        data.edge_index, _ = subgraph(nodes_to_keep_indices, data.edge_index, relabel_nodes=True)
        data.y = node_circle_matrix[nodes_to_keep_mask]
        if data.x is not None:
            data.x = data.x[nodes_to_keep_mask]

                #* v REMOVE ISOLATED v
        data.edge_index, _, not_isolated_nodes_mask = remove_isolated_nodes(data.edge_index)
        data.y = data.y[not_isolated_nodes_mask]
        if data.x is not None:
            data.x = data.x[not_isolated_nodes_mask]

        if remove_data_feats:
            data.x = None
        else: 
            data.feats_data = data.x
            data.x = None
       
                #*  ^   MAKE data.y WITH CIRCLE  ^        

        if verbose:
            G = to_networkx(data, to_undirected=True)

            # Compute the spring layout
            pos = nx.spring_layout(G)

            # Plot the graph
            plt.figure(figsize=(8, 8))
            nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
            plt.title('Graph Visualization with Spring Layout')
            plt.show()

    

    elif dataset_name == 'karate':    
        dataset = KarateClub()

        # Step 1: Get the first graph in the dataset
        data_unsorted = dataset[0]
        data = data_unsorted.clone()

        # Step 2: Sort nodes/features according to sorted indices
        y, sorted_indices = torch.sort(data_unsorted.y)
        data.y = intersecting_tensor_from_non_intersecting_vec(y)
        # Note: data.x in the karate dataset is the number of node so we make it None
        data.x = None

        # Step 3: Update edge_index to reflect new node order
        # Create a mapping from old indices to new indices
        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices.tolist())}

        # Update edge_index using the mapping
        data.edge_index = torch.tensor([[index_mapping[idx.item()] for idx in edge] for edge in data_unsorted.edge_index.t()]).t()

    elif dataset_name == 'imdb':
            # Load the IMDB dataset
        dataset = IMDB(root='path/to/IMDB')
        data = dataset[0]

        # Get the number of nodes for each type
        num_movies = data['movie'].num_nodes
        num_directors = data['director'].num_nodes
        num_actors = data['actor'].num_nodes

        # Create the unified node feature matrix
        x_movie = data['movie'].x
        x_director = data['director'].x
        x_actor = data['actor'].x

        x = torch.cat([x_movie, x_director, x_actor], dim=0)

        # Create one-hot encodings for the labels
        y_movie = torch.tensor([[1, 0, 0]] * num_movies, dtype=torch.float)
        y_director = torch.tensor([[0, 1, 0]] * num_directors, dtype=torch.float)
        y_actor = torch.tensor([[0, 0, 1]] * num_actors, dtype=torch.float)

        # Concatenate to form the final label tensor
        y = torch.cat([y_movie, y_director, y_actor], dim=0)

        # Create the unified edge index
        edge_index_movie_director = data['movie', 'to', 'director'].edge_index
        edge_index_movie_actor = data['movie', 'to', 'actor'].edge_index

        # Adjust the indices
        edge_index_director_movie = data['director', 'to', 'movie'].edge_index + num_movies
        edge_index_actor_movie = data['actor', 'to', 'movie'].edge_index + num_movies + num_directors

        edge_index = torch.cat([edge_index_movie_director, edge_index_director_movie], dim=1)
        edge_index = torch.cat([edge_index, edge_index_movie_actor, edge_index_actor_movie], dim=1)

        # Create the unified train_mask, val_mask, test_mask
        train_mask = torch.cat([data['movie'].train_mask, torch.zeros(num_directors, dtype=torch.bool), torch.zeros(num_actors, dtype=torch.bool)])
        val_mask = torch.cat([data['movie'].val_mask, torch.zeros(num_directors, dtype=torch.bool), torch.zeros(num_actors, dtype=torch.bool)])
        test_mask = torch.cat([data['movie'].test_mask, torch.zeros(num_directors, dtype=torch.bool), torch.zeros(num_actors, dtype=torch.bool)])

        # Create the unified Data object
        data_homogeneous = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        data = data_homogeneous
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)  
        data.x = None  
    
    elif dataset_name == 'amazon':
        data = load_data_('Amazon')
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)

    elif dataset_name == 'reddit':
        data = load_data_('reddit')
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)

    elif dataset_name == 'elliptic':
        data = load_data_('elliptic')
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)
    
    elif dataset_name == 'tfFinance':
        data = load_data_('tfFinance')
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)
    

    elif dataset_name == 'photo':
        data = load_data_('photo')
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)

    elif dataset_name == 'JohnsHopkins55':
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dict = sio.loadmat(os.path.join(current_dir, f"{'facebook100'}/{dataset_name}.mat"))
        adj = sp.coo_matrix(data_dict['A'])
        edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
        raw_attr = sp.lil_matrix(data_dict['local_info'])

        data = Data(edge_index=edge_index, raw_attr=raw_attr, x=None)
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = remove_isolated_nodes(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)

    elif dataset_name == 'Amherst41':
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dict = sio.loadmat(os.path.join(current_dir, f"{'facebook100'}/{dataset_name}.mat"))
        adj = sp.coo_matrix(data_dict['A'])
        edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)

        raw_attr = sp.lil_matrix(data_dict['local_info'])

        data = Data(edge_index=edge_index, raw_attr=raw_attr, x=None)
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = remove_isolated_nodes(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)

    elif dataset_name == 'BlogCatalog':
        data = load_data_matlab_format('anomaly', 'BlogCatalog')
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)
    
    elif dataset_name == 'ACM':
        data = load_data_matlab_format('anomaly', 'ACM')
        data.edge_index = remove_self_loops(data.edge_index)[0]
        data.edge_index = to_undirected(data.edge_index)

    elif dataset_name == 'Flickr':
        data = load_data_matlab_format('anomaly', 'Flickr')
        data.edge_index = remove_self_loops(data.edge_index)[0]
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

    if non_isolated_mask.any():
        if hasattr(data, 'y'):
            if data.y is not None:
                data.y = data.y[non_isolated_mask]
        if hasattr(data, 'gt_nomalous'):
            #? tested?
            if data.gt_nomalous is not None:
                data.gt_nomalous = data.gt_nomalous[non_isolated_mask]
        if hasattr(data, 'train_node_mask'):
            if data.train_node_mask is not None:
                data.train_node_mask = data.train_node_mask[non_isolated_mask]
        if hasattr(data, 'test_normal_mask'):
            if data.test_normal_mask is not None:
                data.test_normal_mask = data.test_normal_mask[non_isolated_mask]
        if hasattr(data, 'test_anomalies_mask'):
            if data.test_anomalies_mask is not None:
                data.test_anomalies_mask = data.test_anomalies_mask[non_isolated_mask]
        if hasattr(data, 'train_idx'):
            if data.train_idx is not None:
                data.train_idx = utils.mask_index(data.train_idx, non_isolated_mask)
        if hasattr(data, 'test_idx'):
            if data.test_idx is not None:
                data.test_idx = utils.mask_index(data.test_idx, non_isolated_mask)
        if hasattr(data, 'val_idx'):
            if data.val_idx is not None:
                data.val_idx = utils.mask_index(data.val_idx, non_isolated_mask)
        if hasattr(data, 'train_normal_idx'):
            if data.train_normal_idx is not None:
                data.train_normal_idx = utils.mask_index(data.train_normal_idx, non_isolated_mask)
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

    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
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