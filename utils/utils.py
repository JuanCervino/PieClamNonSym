import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric import utils
from torch_geometric.utils import dense_to_sparse, to_undirected, remove_self_loops, sort_edge_index, is_undirected, negative_sampling, subgraph, to_networkx, from_networkx, to_dense_adj, k_hop_subgraph, coalesce, contains_self_loops, contains_isolated_nodes, dropout_edge

from torch_geometric.transforms import TwoHop
from networkx.algorithms.cuts import conductance
from torch_geometric.transforms import RemoveDuplicatedEdges
from sklearn.metrics import roc_curve, roc_auc_score
from cutnorm import compute_cutnorm

from utils import utils_pyg as up

from utils.printing_utils import printd


import os
from math import floor
import json
import os
import shutil
import matplotlib.pyplot as plt

#todo: maybe make fit_feat stop before it plateaus

def vanilla_model(model_name):
    if model_name == 'bigclam' or model_name == 'pclam':
        return 'bigclam'
    elif model_name == 'iegam' or model_name == 'piegam':
        return 'iegam'

def prior_model(model_name):
    if model_name == 'bigclam' or model_name == 'pclam':
        return 'pclam'
    elif model_name == 'iegam' or model_name == 'piegam':
        return 'piegam'

def get_edge_probs_from_edges_coords(edges_coords_0, edges_coords_1, lorenz, prior=None, use_prior=False):
    '''given two lists of edges in coordinate shape, get a list of edge probabilities'''
    #* edges_coords_0, edges_coords_1 have shape [N, in_channels]
    dim_feat = edges_coords_0.shape[1]
    if lorenz:
        B = torch.ones(dim_feat//2)
        B = torch.cat([B, -B])
    else:
        B = torch.ones(dim_feat)
    
    B = B.to(edges_coords_0.device)
    fufv = torch.einsum('ij,ij->i', edges_coords_0, B*edges_coords_1)
    if prior and use_prior:
        prior_nodes_0 = torch.exp(prior.forward_ll(edges_coords_0, sum=False))
        prior_nodes_1 = torch.exp(prior.forward_ll(edges_coords_1, sum=False))
        tbr = (1-torch.exp(-fufv)) * prior_nodes_0 * prior_nodes_1
    else:
        tbr = 1-torch.exp(-fufv)
    
    return tbr

def get_prob_graph(x, lorenz, to_sparse=False, prior=None,ret_fufv=False):
    '''given node features, returns the probability graph or the inner product graph'''
    #! need to check this with the thresholds before we continue
    #* x has shape [N, in_channels]
    
    dim_feat = x.shape[1]
    
    if lorenz:
        B = torch.concatenate([torch.ones(dim_feat//2), -torch.ones(dim_feat//2)]).to(x.device)
        fufv = x @ (B*x).T
    else:
        fufv = x @ x.T
    
    if ret_fufv:
        prob_graph = fufv
    else: 
        prob_graph = 1-torch.exp(-fufv)
    
    if prior is not None:
        prior_nodes = torch.exp(prior.forward_ll(x, sum=False))
        prior_of_dyads = prior_nodes.unsqueeze(1) * prior_nodes
        prob_graph = (1-torch.exp(-fufv)) * prior_of_dyads

    prob_graph.fill_diagonal_(0)

    if to_sparse:
        edge_index=dense_to_sparse(prob_graph)[0]
        edge_attr = prob_graph[edge_index[0], edge_index[1]]
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return prob_graph
    # if prior:
    #     #? tested
    #     prior_nodes = torch.exp(prior.forward_ll(x, sum=False))
    #     return (1-torch.exp(-fufv)) * prior_nodes.unsqueeze(1) * prior_nodes
    


def clam_edges_from_feats(points, lorenz):
    ''''sample edges from feature probabilities'''
    if lorenz:
        B = torch.cat([torch.ones(points.shape[1]//2), -torch.ones(points.shape[1]//2)]).unsqueeze(1)
    else:
        B = torch.ones(points.shape[1]).unsqueeze(1)

    prods = torch.matmul(points, B*points.T)
    probs = torch.exp(-prods)

    adj_mat1 = torch.bernoulli(1 - probs)
    #* sample the graph only once since it's undirected:
    adj_mat = torch.triu(adj_mat1)
    edge_index = dense_to_sparse(adj_mat)[0]
    edge_index = to_undirected(edge_index)
    edge_index = remove_self_loops(edge_index)[0]
    edge_index = sort_edge_index(edge_index)

    graph = Data(x=points, edge_index=edge_index)

    return graph


def shuffle_matrix(matrix, perm):
    """Shuffle a matrix according to a given permutation."""
    return matrix[perm, :][:, perm]

def unshuffle_matrix(matrix, perm):
    """Unshuffle a matrix by applying the inverse permutation."""
    inv_perm = torch.argsort(perm)
    return matrix[inv_perm, :][:, inv_perm]

def shuffle_node_feats(node_feats, perm):
    """Shuffle node features according to a given permutation."""
    return node_feats[perm]

def unshuffle_node_feats(node_feats, perm):
    """Unshuffle node features by applying the inverse permutation."""
    inv_perm = torch.argsort(perm)
    return node_feats[inv_perm]


def omit_dyads(data, dyads_to_omit):
        
        ''' this function prepares the data for node ommition. it adds the non edges to omit to the edges array and creates a boolean mask for the edges to omit.
        dyads_to_omit: (edges_to_omit, non_edges_to_omit)'''

        assert len(dyads_to_omit) == 2, 'dyads_to_omit should be a tuple (edges_to_omit, non_edges_to_omit)'

        #! need to assert that the dyads to omit[0] are edges of the graph and that dyads_to_omit[1] are non edges.
        #! need to assert that the dyads to omit are unique
        omitted_dyads_tot = torch.cat([dyads_to_omit[0], dyads_to_omit[1]], dim=1)
        assert is_undirected(omitted_dyads_tot), 'edges in dyads_to_omit should be undirected'
        
        # Check that omitted nodes are unique
        transform = RemoveDuplicatedEdges()
        removed_duplicat = transform(data)
        assert removed_duplicat.edge_index.shape[1] == data.edge_index.shape[1], 'edges in dyads_to_omit should be unique'

        # add the non_edges_to_omit to the edge array
        retained_edges = data.edge_index.clone()
        
        # Add the non edge dyads to the retained edges array
        retained_with_omitted_non_edges = sort_edge_index(torch.cat([retained_edges, dyads_to_omit[1]], dim=1))
        
        # merge the omitted edges to get the total vector of ommitted dyads

        # compare the two edge sets using broadcasting to find the mask.
        retained_with_omitted_non_edges_chub = retained_with_omitted_non_edges.unsqueeze(-1)
        omitted_edges_tot_chub = omitted_dyads_tot.unsqueeze(-1).transpose(1,2)
        eq_tens = (retained_with_omitted_non_edges_chub == omitted_edges_tot_chub).prod(dim=0)
        edge_mask = eq_tens.sum(dim=1)
        #? TESTED that the edge mask has as many elements as in the omitted tensor (8 in the test case)
        #? TESTED that every omitted node appears once
        #? TESTED that retained_with_omitted_non_edges[:,edge_mask.bool()] == omitted_dyads_tot
        
        # change the data object, remember we kept the original edges in edge_index_original
        #! we stopped here, asserting sanity on omitted edges and there is some bug in which things get duplicated.
        data.edge_index = retained_with_omitted_non_edges
        data.edge_attr = edge_mask
    
def several_omits(data, num_sets, percentage_edges, percentage_non_edges=None, same_number=True):
    '''returns a list of lists of dyads to omit. seems unimpotant at the moment since the original omittion is random'''
    if percentage_non_edges is None:
        same_number = True
    num_edges = data.edge_index.shape[1]
    num_non_edges = data.x.shape[0]*(data.x.shape[0]-1) - num_edges
    num_edges_to_omit = floor(percentage_edges*num_edges)
    num_non_edges_to_omit = floor(percentage_non_edges*num_non_edges)
    if same_number:
        num_non_edges_to_omit = num_edges_to_omit
    dyad_sets = []
    for i in range(num_sets):
        dyads_to_omit = get_dyads_to_omit(data.edge_index, floor(num_edges_to_omit), floor(num_non_edges_to_omit))
        dyad_sets.append(dyads_to_omit)
    return dyad_sets


def edge_mask_of_selected_edges(edge_index, edges_to_omit):
    '''returns a boolean mask of the EDGES TO OMIT. if you want the remaining edges, use ~edge_mask_of_selected_edges()'''
    edge_index_chub = edge_index.unsqueeze(-1)
    edges_to_omit_chub = edges_to_omit.unsqueeze(-1).transpose(1,2)
    eq_tens = (edge_index_chub == edges_to_omit_chub).prod(dim=0)
    edge_mask = eq_tens.sum(dim=1)
    return edge_mask.bool()


def sample_edges(edge_index, num_samples):
    '''samples edges from the edge_index'''
    #! problem: it can sample both sides of an edge and then it's not a fixed number of nodes

    canonical_edge_index = torch.stack([
    torch.min(edge_index, dim=0)[0],
    torch.max(edge_index, dim=0)[0]
    ])

    # Remove duplicates by using a unique set of rows
    # Sort each column for consistency
    _, unique_indices = canonical_edge_index.sort(dim=1)
    unique_edges, indices = torch.unique(canonical_edge_index[:, unique_indices[0]], dim=1, return_inverse=True)

    # Sample edges from unique_edges
    indices = torch.randperm(unique_edges.size(1))[:num_samples]
    sampled_edges = unique_edges[:, indices]

    # perm = torch.randperm(edge_index.size(1))

    # Select the first num_samples columns
    # sampled_edge_index = to_undirected(edge_index[:, perm[:num_samples]])
    return to_undirected(sampled_edges)

def get_dyads_to_omit(edge_index, p_sample_edge, p_sample_non_edge=None):
    
    if p_sample_edge == 0:
        return None

    assert p_sample_edge <= 1, 'p_sample_edge should be a probability'

    if p_sample_non_edge is None:
        p_sample_non_edge = p_sample_edge
    num_edges = edge_index.shape[1]

    # sampled_edge_index = sample_edges(edge_index, num_samples_edge)
    # sampled_non_edge_index = sample_edges(non_edge_index, num_samples_non_edge)
    
    edge_index_rearanged, edge_mask_retain = up.edge_mask_drop_and_rearange(edge_index, p_sample_edge)
    
    sampled_edge_index = edge_index_rearanged[:, ~edge_mask_retain]
    sampled_non_edge_index = sort_edge_index(negative_sampling(
                            edge_index, 
                            num_neg_samples=floor(num_edges*p_sample_non_edge), 
                            force_undirected=True))

    
    dyads_to_omit = (sampled_edge_index, sampled_non_edge_index, edge_index_rearanged, edge_mask_retain)
    
    #? TESTED that edge_index_rearanged contains the same edges as edge_index. 
    #? TESTED edge_index_rearanged[: , ~edge_mask_retain] == sampled_edge_index
    return dyads_to_omit



def edges_by_coords(data):
    '''
    given edges and node features, returns the edges as pairs of feature vectors
    '''

    #* edges_coords_i is the features of the left/right side of the edge pair
    edges_coords_0 = data.x[data.edge_index[0, :], :]
    edges_coords_1 = data.x[data.edge_index[1, :], :]
    return edges_coords_0, edges_coords_1


def background_prob(graph):
    epsilon = (graph.edge_index.shape[1])/(graph.x.shape[0]*(graph.x.shape[0]-1))
    delta = np.sqrt(-np.log(1 - epsilon))
    return delta

def mask_from_node_list(node_list, num_nodes):
    mask = torch.zeros(num_nodes).bool()
    mask[node_list] = True
    return mask

def delete_file_by_str(file_path, str_to_find):

    for root, dirs, files in os.walk(file_path):
        for dir in dirs:
            if str_to_find in dir:
                shutil.rmtree(os.path.join(root, dir))
    


def save_feats_prior_hypers(x, prior, configs_dict, folder_path, overwrite=False):
    base_dir_path = '../checkpoints/' + folder_path
    dir_path = base_dir_path
    if not overwrite:
        i=1
        while os.path.exists(dir_path):
            dir_path = f"{base_dir_path}_{i}"
            i += 1
    os.makedirs(dir_path, exist_ok=True)

    x_to_save = x.clone()
    
    torch.save(x_to_save, dir_path + '/x.pth')
    if prior is not None:
        prior.save_weights(dir_path)
    with open(dir_path + '/config.json', 'w') as f:
        json.dump(configs_dict, f, indent=4)


def save_feats_edges_prior_hypers(data, prior, config_dict, folder_path, overwrite=False):
    base_dir_path = '../checkpoints/' + folder_path
    dir_path = base_dir_path
    if not overwrite:
        i=1
        while os.path.exists(dir_path):
            dir_path = f"{base_dir_path}_{i}"
            i += 1
    os.makedirs(dir_path, exist_ok=True)

    data_to_save = data.clone()

    torch.save(data_to_save, dir_path + '/data.pth')
    if prior is not None:
        prior.save_weights(dir_path)
    with open(dir_path + '/config.json', 'w') as f:
        json.dump(config_dict, f, indent=4)



def extremes(lst):
    return [min(lst), max(lst)]

def plot_roc_curve(fpr, tpr, thresholds):
    '''false positive rate on the x axis'''
    fpr_tpr_vec = torch.cat((torch.tensor(fpr).view(-1, 1), torch.tensor(tpr).view(-1, 1)), dim=1)
    dists_from_11 = torch.sqrt(((torch.tensor([0,1]) - fpr_tpr_vec)**2).sum(dim=1))
    min_dist_idx = torch.argmin(dists_from_11)
    min_dist = dists_from_11[min_dist_idx]
    best_threshold = thresholds[min_dist_idx]

    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.scatter(*fpr_tpr_vec[min_dist_idx].detach().numpy(), color='red', label='closest point to (0,1)')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return min_dist, best_threshold



def roc_of_omitted_dyads(x, lorenz, dyads_to_omit=None, prior=None, use_prior=False, verbose=False):
    '''calculates the minimun distance from 0,1 in the roc curve and the auc. mathematically there is no sense in using the prior'''
    if dyads_to_omit is None:
        return {'auc': 0.0}
    edges_coords_0, edges_coords_1 = edges_by_coords(Data(x=x, edge_index=dyads_to_omit[0]))

    non_edges_coords_0, non_edges_coords_1 = edges_by_coords(Data(x=x, edge_index=dyads_to_omit[1]))

    edge_probs = get_edge_probs_from_edges_coords(edges_coords_0, edges_coords_1, lorenz, prior, use_prior)
    non_edge_probs = get_edge_probs_from_edges_coords(non_edges_coords_0, non_edges_coords_1, lorenz, prior, use_prior)
    
    y_true = torch.cat((torch.ones(len(edge_probs)), torch.zeros(len(non_edge_probs)))).cpu().detach().numpy()
    y_score = torch.cat((edge_probs, non_edge_probs)).cpu().detach().numpy()
    fpr, tpr, thresholds = roc_curve(y_true, y_score)   
    auc = roc_auc_score(y_true, y_score)

    fpr_tpr_vec = torch.cat((torch.tensor(fpr).view(-1, 1), torch.tensor(tpr).view(-1, 1)), dim=1)
    dists_from_11 = torch.sqrt(((torch.tensor([0,1]) - fpr_tpr_vec)**2).sum(dim=1))
    min_dist_idx = torch.argmin(dists_from_11)
    min_dist = dists_from_11[min_dist_idx]
    best_threshold = thresholds[min_dist_idx]


    if verbose:
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve')
        plt.scatter(*fpr_tpr_vec[min_dist_idx].detach().numpy(), color='red', label='closest point to (0,1)')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    return {'min_dist':min_dist.item(), 'auc':auc, 'best_thresh':best_threshold}

# 88 88b 88 88 888888 
# 88 88Yb88 88   88   
# 88 88 Y88 88   88   
# 88 88  Y8 88   88   

#! i think that maybe the problem with anomaly planting is that the anomaly set can be larger than a community.
def k_minimal_neighborhoods(data, k):   
    '''returns the k minimal neighborhoods of the graph'''
    # if not hasattr(data, 'node_mask') or (hasattr(data, 'node_mask') and data.node_mask is None):
    #     node_mask = torch.ones(data.num_nodes).bool()
    # else:
    #     node_mask = data.node_mask

    G = to_networkx(data)
    neighborhoods = []
    conductances = []
    node_list = list(range(data.num_nodes))
    
    for node in node_list:
        neighbors = list(G.neighbors(node))
        try:
            conductances.append(conductance(G, neighbors))
            neighborhoods.append(neighbors)
        except ZeroDivisionError as e:
            '''if a node is isolated it will raise a zero division error. in this case we do not count it's empty neighborhood as a minimal neighborhood.'''
            a=0

    neighborhoods = [x for _, x in sorted(zip(conductances, neighborhoods))]
    minimal_neighborhoods = neighborhoods[:k]
    return minimal_neighborhoods

# 8888b.  88 .dP"Y8 888888    db    88b 88  dP""b8 888888 
#  8I  Yb 88 `Ybo."   88     dPYb   88Yb88 dP   `" 88__   
#  8I  dY 88 o.`Y8b   88    dP__Yb  88 Y88 Yb      88""   
# 8888Y"  88 8bodP'   88   dP""""Yb 88  Y8  YboodP 888888 


def cut_log_data(data, lorenz, d=0.2, approx_method='sdp', return_d=False, verbose=False):
    
    '''implementation of the cut log distance from the paper.
    incentive in choosing d: chose d such that 1-d is the probability of the edge being in the graph.'''
    
    assert d <= 1 or d>0 , 'in cut log distance, d should be in (0,1].'
    if approx_method not in ['sdp', 'rounded']:
        raise ValueError('in cutnorm: approx_method should be either sdp or rounded')
    p = get_prob_graph(data.x, lorenz=lorenz, to_sparse=False).cpu().numpy()
    if hasattr(data, 'sbm'):
        q = data.sbm.cpu().numpy()
    else:
        q = (to_dense_adj(data.edge_index)[0]*(1 - d)).cpu().numpy()

    inverted_p = np.log(1 - p)
    inverted_q = np.log(1 - q)
    # i want to see what the probabilities of bigclam by playing with d see that 
    cutn_round, cutn_sdp, info = compute_cutnorm(inverted_p, inverted_q)
    if return_d:
        tbr = cutn_round + d if approx_method == 'rounded' else cutn_sdp + d
    else:
        tbr = cutn_round if approx_method == 'rounded' else cutn_sdp 
    
    return tbr
    



def cut_distance_data(data, lorenz, approx_method='sdp', verbose=False):
    '''calculates the cutnorm between the edge index and  the affiliation features x'''
    if approx_method not in ['sdp', 'rounded']:
        raise ValueError('in cutnorm: approx_method should be either sdp or rounded')
    prob_adj = get_prob_graph(data.x, lorenz=lorenz, to_sparse=False)
    if hasattr(data, 'sbm'):
        gt_adj = data.sbm
    else:
        gt_adj = to_dense_adj(data.edge_index)[0]
    # Generate Erdos Renyi Random Graph (Simple/Undirected)

    # Compute l1 norm

    # Compute cutnorm
    cutn_round, cutn_sdp, info = compute_cutnorm(prob_adj.clone().cpu().numpy(), gt_adj.clone().cpu().numpy())
    if verbose:
        printd(f"\ncutnorm rounded: {cutn_round}")  # prints cutnorm rounded solution near ~0
        printd(f"\ncutnorm sdp: {cutn_sdp}")  # prints cutnorm sdp solution near ~0

    return cutn_round if approx_method == 'rounded' else cutn_sdp

def l2_distance_data(data, lorenz, verbose=False):
    prob_adj = get_prob_graph(data.x, lorenz=lorenz, to_sparse=False)
    gt_adj = to_dense_adj(data.edge_index)[0]
    l2_dist = torch.norm(prob_adj - gt_adj, p=2)
    if verbose:
        printd(f"\nl2 distance: {l2_dist}")
    return l2_dist

def relative_l2_distance_data(data, lorenz, verbose=False):
    prob_adj = get_prob_graph(data.x, lorenz=lorenz, to_sparse=False)
    gt_adj = to_dense_adj(data.edge_index)[0]
    rel_l2_dist = torch.norm(prob_adj - gt_adj, p=2)/torch.norm(gt_adj, p=2)
    # if verbose:
    #     printd(f"\nl2 distance: {rel_l2_dist}")
    return rel_l2_dist

def get_fat_ds_test_OLD(ds, train_node_mask, test_node_mask):
    '''densifies the dataset with the test addition such that the subgraph of the train set is the same as in training.
    in here think of adding and removing edges'''

    train_node_index = torch.where(train_node_mask)[0]
    test_node_index = torch.where(test_node_mask)[0]

    # DENSIFY TRAIN EDGES
    train_edge_index, train_edge_attr = subgraph(
                            train_node_index, 
                            ds.edge_index, 
                            ds.edge_attr, 
                            relabel_nodes=False, 
                            num_nodes=ds.num_nodes)
    
    train_data = Data(
                edge_index=train_edge_index, 
                edge_attr=train_edge_attr, 
                num_nodes=ds.num_nodes)
    
    fat_train_data = TwoHop()(train_data)
    fat_train_data.edge_attr = torch.ones_like(fat_train_data.edge_attr)
    # =============================================================
    
    # DENSIFY TEST EDGES (take test set and the 1 hop nodes)
    _, test_edge_index, _, train_edge_mask = k_hop_subgraph(
                                    test_node_index, 
                                    1,
                                    ds.edge_index,  
                                    relabel_nodes=False)
    
    test_edge_attr = ds.edge_attr[train_edge_mask]
    test_data = Data(
                edge_index=test_edge_index, 
                edge_attr=test_edge_attr, 
                num_nodes=ds.num_nodes)
    
    fat_test_data = TwoHop()(test_data)
    fat_test_data.edge_attr = torch.ones_like(fat_test_data.edge_attr)
    # =================================================================

    # REMOVE OVERLAPPING EDGES FROM DENSE TEST
    _, edges_rogue_fat_test, _, edge_mask = k_hop_subgraph(
                                train_node_index, 
                                0, 
                                fat_test_data.edge_index, 
                                relabel_nodes=False, 
                                num_nodes=ds.num_nodes)
                    
    fat_test_data.edge_index = fat_test_data.edge_index[:, ~edge_mask]
    fat_test_data.edge_attr = fat_test_data.edge_attr[~edge_mask]
    # =================================================================
    # todo: ron's suggestion was to make the densification in the test set only through the train nodes so that if two normals are connected through an anomaly they won't get connected in the test set. the densification happens only through train nodes. 
    
    

    # COMBINE TRAIN AND TEST EDGES
    fat_ds_with_anomalies = ds.clone()
    fat_ds_with_anomalies.edge_index = torch.cat(
                [fat_train_data.edge_index, fat_test_data.edge_index], 
                dim=1)
    
    fat_ds_with_anomalies.edge_attr = torch.cat(
                [fat_train_data.edge_attr, fat_test_data.edge_attr], 
                dim=0)
    
    fat_ds_with_anomalies.edge_index, fat_ds_with_anomalies.edge_attr = sort_edge_index(
                fat_ds_with_anomalies.edge_index, 
                fat_ds_with_anomalies.edge_attr, 
                fat_ds_with_anomalies.num_nodes)
    # =================================================================

    # ASSERTIONS
    assert contains_self_loops(fat_ds_with_anomalies.edge_index) == False , 'self loops in fat_ds_with_anomalies'
    assert is_undirected(fat_ds_with_anomalies.edge_index), 'fat_ds is directed!'
    assert contains_isolated_nodes(fat_ds_with_anomalies.edge_index, fat_ds_with_anomalies.num_nodes) == False, 'isolated nodes in fat_ds_with_anomalies'
    # ============================================================
    return fat_ds_with_anomalies
    

def densify_ds_via_train_nodes(ds, train_node_mask, test_node_mask):
    '''densification through training nodes'''
    #algo: (imagine line separating train and test, graph expands and contracts...)
    # 1. densify train 

    # 2. tau = 1-hop from train nodes
    # 3. tau(edges) = tau(edges) - test subgraph (edges)
    # 4. densify tau (all of the additional edges to the test are here)

    # remove rogue train edges:

    # 5. xi = 1-hop of from test nodes in tau 
    # 6. remove the train subgraph from xi.   
    # 7. cat and coalesce original test, densified test, densified train
    assert torch.all(train_node_mask ^ test_node_mask), 'densify_ds_from_train: train and test nodes overlap'
    assert (ds.edge_attr == 1).all(), 'densify_ds_from_train: edge attributes are not all 1 which means you omitted dyads before densification'

    train_node_index = torch.where(train_node_mask)[0]
    test_node_index = torch.where(test_node_mask)[0]

    # 1. DENSIFY TRAIN EDGES
    _, train_edge_index, _, train_edge_mask = k_hop_subgraph(
                            train_node_index,
                            0, 
                            ds.edge_index, 
                            relabel_nodes=False, 
                            num_nodes=ds.num_nodes)
    
    train_edge_attr = ds.edge_attr[train_edge_mask]
    
    train_data = Data(
                edge_index=train_edge_index, 
                edge_attr=train_edge_attr, 
                num_nodes=ds.num_nodes)
    
    fat_train_data = TwoHop()(train_data)
    fat_train_data.edge_attr = torch.ones_like(fat_train_data.edge_attr)
    # =============================================================
    
    # DENSIFY TEST EDGES 
     
            # 2. tau: 1-hop train 
    _, tau_edge_index, _, tau_edge_mask = k_hop_subgraph(
                                    train_node_index, 
                                    1,
                                    ds.edge_index,  
                                    relabel_nodes=False,
                                    num_nodes=ds.num_nodes)
    
    tau_edge_attr = ds.edge_attr[tau_edge_mask]
    tau_data = Data(
                edge_index=tau_edge_index, 
                edge_attr=tau_edge_attr, 
                num_nodes=ds.num_nodes)
    
            #  3. remove test subgraph from tau before densification
    _, test_edges_remove_from_tau, _, test_edge_mask_remove_from_tau = k_hop_subgraph(
                                test_node_index, 
                                0, 
                                tau_edge_index, 
                                relabel_nodes=False, 
                                num_nodes=ds.num_nodes)
                    
    tau_data.edge_index = tau_data.edge_index[:, ~test_edge_mask_remove_from_tau]
    tau_data.edge_attr = tau_data.edge_attr[~test_edge_mask_remove_from_tau]
    
            # 4. densify (without test edges) 
                         #(maybe don't need entire train graph, only use xi...)
    fat_tau_data = TwoHop()(tau_data)
    #! don't know if the next one is a good idea.... might as well just do it at the end
    fat_tau_data.edge_attr = torch.ones_like(fat_tau_data.edge_attr)

            # 5. xi: 1-hop from test in tau (remove densification from test in train)
    _, fat_xi_edge_index, _, fat_xi_edge_mask = k_hop_subgraph(
                                    test_node_index, 
                                    1,
                                    fat_tau_data.edge_index,  
                                    relabel_nodes=False,
                                    num_nodes=ds.num_nodes)
    
    
    fat_xi_edge_attr = fat_tau_data.edge_attr[fat_xi_edge_mask]
    fat_xi_data = Data(
                edge_index=fat_xi_edge_index, 
                edge_attr=fat_xi_edge_attr, 
                num_nodes=ds.num_nodes)
    
            # 6. remove train subgraph from xi
    
    _, rogue_train_edges_fat_test, _, rogue_train_edge_mask = k_hop_subgraph(
                                train_node_index, 
                                0, 
                                fat_xi_edge_index, 
                                relabel_nodes=False, 
                                num_nodes=ds.num_nodes)
                    
    fat_xi_data.edge_index = fat_xi_data.edge_index[:, ~rogue_train_edge_mask]
    fat_xi_data.edge_attr = fat_xi_data.edge_attr[~rogue_train_edge_mask]
            
            # 7. combine xi and the test subgraph (and coalesce - remove redundant edges)
    _, test_edge_index, _, test_edge_mask = k_hop_subgraph(
                                    test_node_index,
                                    0,
                                    ds.edge_index,
                                    relabel_nodes=False)
    
    test_edge_attr = ds.edge_attr[test_edge_mask]

    fat_test_edge_index = coalesce(torch.cat([fat_xi_data.edge_index, test_edge_index], dim=1))
    #! warning: do densification before omit edges for now!
    fat_test_edge_attr = torch.ones(fat_test_edge_index.shape[1]).to(fat_test_edge_index.device)
    # ======================================================================
    #      (end making test set densification) #? varified with dummy test

    # COMBINE TRAIN AND TEST EDGES
        # all edges between test nodes

    fat_ds_with_anomalies = ds.clone()
    fat_ds_with_anomalies.edge_index = torch.cat(
                [fat_train_data.edge_index, fat_test_edge_index], 
                dim=1)
    # there will be duplicity in the edges, see that it's in the test set
    
    fat_ds_with_anomalies.edge_attr = torch.cat(
                [fat_train_data.edge_attr, fat_test_edge_attr], 
                dim=0)
    
    #? test that there is duplicity but only in edges between test nodes
    fat_ds_with_anomalies.edge_index, fat_ds_with_anomalies.edge_attr = sort_edge_index(
                fat_ds_with_anomalies.edge_index, 
                fat_ds_with_anomalies.edge_attr, 
                fat_ds_with_anomalies.num_nodes)
    # =================================================================

    # ASSERTIONS
    assert contains_self_loops(fat_ds_with_anomalies.edge_index) == False , 'self loops in fat_ds_with_anomalies'
    assert is_undirected(fat_ds_with_anomalies.edge_index), 'fat_ds is directed!'
    assert contains_isolated_nodes(fat_ds_with_anomalies.edge_index, fat_ds_with_anomalies.num_nodes) == False, 'isolated nodes in fat_ds_with_anomalies'
    # ============================================================
    return fat_ds_with_anomalies
    
def mask_to_index(mask):
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask)
    index = torch.where(mask)[0]
    return index

def index_to_mask(index, num_elements):
    if not isinstance(index,torch.Tensor):
        index = torch.tensor(index)
    mask = torch.zeros(num_elements, dtype=torch.bool)
    mask[index] = True
    return mask

def mask_index(index, mask):
    #? testes
    num_elements = len(mask)
    i2m = index_to_mask(index, num_elements)
    new_mask = i2m[mask]
    new_index = mask_to_index(new_mask)
    return new_index

def scheduler_step(scheduler, optimizer, feat_params, prior_params, verbose=False):

    old_optimizer_lr = optimizer.param_groups[0]['lr'] #! debug
    scheduler.step()
    new_optimizer_lr = optimizer.param_groups[0]['lr'] #! debug
    scheduler_updated = False
    if scheduler.last_epoch%scheduler.step_size == 0:
        scheduler_updated = True
        assert(old_optimizer_lr != new_optimizer_lr), f'optimizeer condition not met' #! debug for a while
        old_feats_lr = feat_params['lr'] 
        old_feats_n_iter = feat_params['n_iter']
        feat_params['lr'] = max(feat_params['lr']/2, 0.00000001)
        old_noise_amp = prior_params['noise_amp'] 
        prior_params['lr'] = new_optimizer_lr
        prior_params['noise_amp'] = max(prior_params['noise_amp']/2, 0.0001)
        if verbose:
            printd(f'\nscheduler made step. changes:') 
            print(f'feats lr: {old_feats_lr} to {feat_params["lr"]}')  
            print(f'feats n_iter changed from {old_feats_n_iter} to {feat_params["n_iter"]}') 
            print(f'noise_amp from {old_noise_amp} to {prior_params["noise_amp"]}') 
            print(f'prior lr from {old_optimizer_lr} to {optimizer.param_groups[0]["lr"]}') 

    return scheduler_updated