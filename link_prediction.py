import torch
from torch_geometric import utils as upg
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import math

from utils import utils
from utils import utils_pyg as up


def get_dyads_to_omit_old(edge_index, p_sample_edge, p_sample_non_edge=None):
    
    if p_sample_edge == 0:
        return None

    assert p_sample_edge <= 1, 'p_sample_edge should be a probability'

    #todo: omit dyads only from the dyads that have edge attr
    if p_sample_non_edge is None:
        p_sample_non_edge = p_sample_edge
    num_edges = edge_index.shape[1]

    # sampled_edge_index = sample_edges(edge_index, num_samples_edge)
    # sampled_non_edge_index = sample_edges(non_edge_index, num_samples_non_edge)
    
    edge_index_rearanged, edge_mask_retain = up.edge_mask_drop_and_rearange(edge_index, p_sample_edge)
    
    sampled_edge_index = edge_index_rearanged[:, ~edge_mask_retain]
    sampled_non_edge_index = upg.sort_edge_index(upg.negative_sampling(
                            edge_index, 
                            num_neg_samples=math.floor(num_edges*p_sample_non_edge), 
                            force_undirected=True))

    
    dyads_to_omit = (sampled_edge_index, sampled_non_edge_index, edge_index_rearanged, edge_mask_retain)
    
    #? TESTED that edge_index_rearanged contains the same edges as edge_index. 
    #? TESTED edge_index_rearanged[: , ~edge_mask_retain] == sampled_edge_index
    return dyads_to_omit


def get_dyads_to_omit(
          edge_index, 
          edge_attr, 
          p_sample_edge, 
          p_sample_non_edge=None, 
        #   omitted_previously=(torch.empty(2, 0), torch.empty(2, 0))
          ):
    '''
    algo: 
    general idea: create 4 sets: A(omitted dyads) = {were 0 in the beginning}, B(retained edges) = {were 1 in the beginning and are still 1}, C(omitted_edges) = {were 1 in the beginning and are now 0}, D(omitted non edges) = {sampled non edges that are 0}

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
        p_sample_non_edge = p_sample_edge
    
    # 1. split the edges into attr 1 = (B or C) and attr 0 = A.
    B_or_C = edge_index[:, edge_attr]
    A = edge_index[:, ~edge_attr]
    assert utils.is_undirected(B_or_C), 'B_or_C should be undirected'
    assert utils.is_undirected(A), 'A should be undirected'
    
    # 2. sample edges from (B or C) (this also rearanges them) - creating B and C.
    B_or_C_rearanged, edge_mask_retain = up.edge_mask_drop_and_rearange(B_or_C, p_sample_edge)
    B = B_or_C_rearanged[:, edge_mask_retain]
    C = B_or_C_rearanged[:, ~edge_mask_retain]

    # 3. sample D from the non edges using negative sampling.
    num_edges = edge_index.shape[1]
    D = upg.sort_edge_index(upg.negative_sampling(
                            edge_index, 
                            num_neg_samples=math.floor(num_edges*p_sample_non_edge), 
                            force_undirected=True))
    
    
    
    edge_index_rearanged = torch.cat([B, C, D, A], dim=1)
    # edge_mask_retain will be the edge attr
    edge_attr_rearanged = torch.cat([torch.ones(B.shape[1]), torch.zeros(C.shape[1]), torch.zeros(D.shape[1]), torch.zeros(A.shape[1])]).bool()

    
    dyads_to_omit = ((C, D), edge_index_rearanged, edge_attr_rearanged)
    
    return dyads_to_omit


def omit_dyads(data, dyads_to_omit):
        
        ''' this function prepares the data for dyad ommition. it adds the non edges to omit to the edges array and creates a boolean mask for the edges to omit.
        dyads_to_omit: (edges_to_omit, non_edges_to_omit). dropped dyads get the edge attr 0 and the retained edges get the edge attr 1.
        PARAM: dyads_to_omit: tuple 4 elements:'''
            
        
        
        assert len(dyads_to_omit) == 4, 'dyads_to_omit should be a tuple (edges_to_omit, non_edges_to_omit, edge_index_rearanged, edge_mask_rearanged)'
        assert dyads_to_omit[2].shape[1] == data.edge_index.shape[1], 'dyads_to_omit[2] should be the same as self.data.edge_index but rearanged'
        assert utils.coalesce(dyads_to_omit[2]).shape[1] == data.edge_index.shape[1], 'dyads_to_omit[2] should be the same as self.data.edge_index but rearanged'



        omitted_dyads_tot = torch.cat([dyads_to_omit[0], dyads_to_omit[1]], dim=1)
        
        rearanged_edge_index = dyads_to_omit[2]
        rearanged_edge_index_with_omitted_non_edges = torch.cat([rearanged_edge_index, dyads_to_omit[1]], dim=1)
        edge_attr = torch.cat([dyads_to_omit[3], torch.zeros(dyads_to_omit[1].shape[1]).bool()])
        assert upg.is_undirected(rearanged_edge_index_with_omitted_non_edges), 'edges in dyads_to_omit should be undirected'
        assert (rearanged_edge_index_with_omitted_non_edges[:, ~edge_attr] == omitted_dyads_tot ).all(), 'edge_attr should be 0 for omitted dyads'
        assert rearanged_edge_index_with_omitted_non_edges.shape[1] == data.edge_index.shape[1] + dyads_to_omit[1].shape[1], 'rearanged_edge_index_with_omitted_non_edges should have the same number of edges as the original edge_index + the non edges to omit'
        # so edge_attr == 0 for omitted edges and ==1 for non omitted

        return rearanged_edge_index_with_omitted_non_edges, edge_attr



def omit_densify_split(
          data, 
          dyads_to_omit, 
          val_rel_size=0.0, 
          densify=False,
          clone=True):
        '''this function omits dyads from the data and densifies it. it also splits the omitted dyads into validation and test sets. the validation split is not used as the test set is split separately.
        '''
        if clone:
            data_clone = data.clone()
        else:
            data_clone = data
        #todo: make omit dyads handle a dataset that's already been omitted.
        data_clone.edge_index, data_clone.edge_attr = omit_dyads(data_clone, dyads_to_omit)
        #todo: densify: just do two hop densification and the attr for a densified node should be 1

        if densify:
            data_clone.edge_index, data_clone.edge_attr = up.my_two_hop(data_clone)

        omitted_val = ([dyads_to_omit[0][:, :round(dyads_to_omit[0].shape[1]*val_rel_size)], dyads_to_omit[1][:, :round(dyads_to_omit[1].shape[1]*val_rel_size)]])

        omitted_test = ([dyads_to_omit[0][:, round(dyads_to_omit[0].shape[1]*val_rel_size):], dyads_to_omit[1][:, round(dyads_to_omit[1].shape[1]*val_rel_size):]])

        data_clone.omitted_dyads = (omitted_test, omitted_val)
        return data_clone
    




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
