import torch
from torch_geometric import utils
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import math
import os


from utils import utils
from utils import utils_pyg as up





def get_dyads_to_omit(
          edge_index, 
          edge_attr, 
          p_sample_edge, 
          p_sample_non_edge=None, 
        #   omitted_previously=(torch.empty(2, 0), torch.empty(2, 0))
          ):
    '''
    the edges that have attr 0 are omitted (edges and non edges). the non edges that are omitted are inserted into the edge index and also given attr 0.
    algo: 
    general idea: create 4 sets: A(pre-omitted dyads) = {were 0 in the beginning}, B(retained edges) = {were 1 in the beginning and are still 1}, C(newly omitted_edges) = {were 1 in the beginning and are now 0}, D(newly omitted non edges) = {sampled non edges that have attr are 0}

    1. split the edges into attr 1 = (B or C) and attr 0 = A.
    2. sample edges from (B or C) (this also rearanges them) - creating B and C.
    3. sample D from the non edges using negative sampling.
    4. concatenate [B, C, A, D] to get the new edge index, in this way the 0 attr edges are also arranged [edges, non edges]
    5. return (C, D, new_edge_index, edge_attr)
    
    '''
    # 0. pre-processing
    if p_sample_edge == 0:
        return ((torch.empty(2, 0), torch.empty(2, 0)), edge_index, edge_attr)
   
    assert p_sample_edge <= 1, 'p_sample_edge should be a probability'
    
    if p_sample_non_edge is None:
        # factor of 5 to replicate the paper.
        p_sample_non_edge = 5*p_sample_edge
    
    # 1. split the edges into attr 1 = (B or C) and attr 0 = A.
    B_or_C = edge_index[:, edge_attr]
    A = edge_index[:, ~edge_attr]
    assert utils.is_undirected(B_or_C), 'B_or_C should be undirected'
    assert utils.is_undirected(A), 'A should be undirected'
    
    # 2. sample edges from (B or C) (this also rearanges them) - creating B and C.
    #todo: where can i replace the sampling with existing dyads to omit
    B_or_C_rearanged, edge_mask_retain = up.edge_mask_drop_and_rearange(B_or_C, p_sample_edge)
    B = B_or_C_rearanged[:, edge_mask_retain]
    C = B_or_C_rearanged[:, ~edge_mask_retain]

    # 3. sample D from the non edges using negative sampling.
    num_edges = edge_index.shape[1]
    D = utils.sort_edge_index(utils.negative_sampling(
                            edge_index, 
                            num_neg_samples=math.floor(num_edges*p_sample_non_edge), 
                            force_undirected=True))
    
    
    
    edge_index_rearanged = torch.cat([B, C, D, A], dim=1)
    # edge_mask_retain will be the edge attr
    edge_attr_rearanged = torch.cat([torch.ones(B.shape[1]), torch.zeros(C.shape[1]), torch.zeros(D.shape[1]), torch.zeros(A.shape[1])]).bool()

    
    dyads_to_omit = ((C, D), edge_index_rearanged, edge_attr_rearanged)
    
    return dyads_to_omit



def omit_dyads(
          edge_index, 
          edge_attr, 
          dyads_to_omit 
          ):
    '''
    You are given a set of dyads to omit from a dataset. you want to see which of the dyads is an edge in the datset and which is a non edge. the non edges you want to add to the dataset with attr 0 and the edges you want to turn to 0

    algo: 
    for each of the dyads, search if it exists in the dataset. If it exists, turn the attr to 0, if it doesn't exist, turn the attr to 1
    A - pre omitted dyads with attr 0
    B - retained edges
    C - edges to omit
    D - non edges to omit
    '''
    #todo: need to assert that dyads_to_omit[0] is in edge_index and that dyads_to_omit[1] is not at all in edge_index
    # Assert that dyads_to_omit[0] (C) is in edge_index
    C = dyads_to_omit[0]
    D = dyads_to_omit[1]
    
    C_set = set(map(tuple, C.t().tolist()))
    D_set = set(map(tuple, D.t().tolist()))
    edge_index_set = set(map(tuple, edge_index.t().tolist()))
    assert C_set.issubset(edge_index_set), "dyads_to_omit[0] must be a subset of edge_index"
    assert D_set.isdisjoint(edge_index_set), "dyads_to_omit[1] must not overlap with edge_index"

    
    A = edge_index[:, ~edge_attr]
    
    # B is the retaied edges so i need to get the set of 
    B_or_C = edge_index[:, edge_attr]
    assert utils.is_undirected(B_or_C), 'B_or_C should be undirected'
    assert utils.is_undirected(A), 'A should be undirected'
    

    B_or_C_set = set(map(tuple, B_or_C.t().tolist()))
    B_set = B_or_C_set - C_set
    B = torch.tensor(list(B_set), dtype=torch.long).t()
 
    edge_index_rearanged = torch.cat([B, C, D, A], dim=1)
    # edge_mask_retain will be the edge attr
    edge_attr_rearanged = torch.cat([torch.ones(B.shape[1]), torch.zeros(C.shape[1]), torch.zeros(D.shape[1]), torch.zeros(A.shape[1])]).bool()

    
    dyads_to_omit = ((C, D), edge_index_rearanged, edge_attr_rearanged)
    
    return dyads_to_omit


#todo: here add another function for when we need to use the ogb evaluator. 
#todo: the ogb eveluator takes a list of positive predictions and negative predictions. they should be the 

def ogb_hAk_omitted_dyads(x, lorenz, dyads_to_omit=None, prior=None, use_prior=False):
    '''returns the arrays of probabilities for both edges and non edges from the omit set'''
    #! dyads to omit need to be both for test and for val
    # evaluator = Evaluator(name='ogbl-ddi')
    
    if dyads_to_omit is None:
        return {'ogb_hAk': 0.0}
    
    # get the edges by coords of the dyads to omit right now i just get the dyads to omit
    edges_coords_0, edges_coords_1 = utils.edges_by_coords(Data(x=x, edge_index=dyads_to_omit[0]))

    non_edges_coords_0, non_edges_coords_1 = utils.edges_by_coords(Data(x=x, edge_index=dyads_to_omit[1]))

    edge_probs = utils.get_edge_probs_from_edges_coords(edges_coords_0, edges_coords_1, lorenz, prior, use_prior)
    non_edge_probs = utils.get_edge_probs_from_edges_coords(non_edges_coords_0, non_edges_coords_1, lorenz, prior, use_prior)

    # the following arrays are identical to the valid/test splits
    edge_probs_ogb = edge_probs[:edge_probs.shape[0]//2]
    non_edge_probs_ogb = non_edge_probs[:non_edge_probs.shape[0]//2]
    return edge_probs_ogb, non_edge_probs_ogb 
    

#todo: can i still do it if i change the edge_attr when loading the dataset?
def roc_of_omitted_dyads(x, lorenz, dyads_to_omit=None, prior=None, use_prior=False, verbose=False):
    '''calculates the minimun distance from 0,1 in the roc curve and the auc. mathematically there is no sense in using the prior'''
    if dyads_to_omit is None:
        return {'auc': 0.0}
    
    edges_coords_0, edges_coords_1 = utils.edges_by_coords(Data(x=x, edge_index=dyads_to_omit[0]))

    non_edges_coords_0, non_edges_coords_1 = utils.edges_by_coords(Data(x=x, edge_index=dyads_to_omit[1]))

    edge_probs = utils.get_edge_probs_from_edges_coords(edges_coords_0, edges_coords_1, lorenz, prior, use_prior)
    non_edge_probs = utils.get_edge_probs_from_edges_coords(non_edges_coords_0, non_edges_coords_1, lorenz, prior, use_prior)
    
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
