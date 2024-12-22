import torch
from torch_geometric.utils import to_dense_adj, sort_edge_index, negative_sampling, to_undirected, is_undirected
from torch_geometric.data import Data
import sys
if '..' not in sys.path:
    sys.path.insert(0, '..')
from trainer import Trainer
from utils.utils import edge_mask_of_selected_edges, sample_edges, densify_ds_via_train_nodes, get_fat_ds_test_OLD
from utils import utils
from utils.plotting import *
from utils.printing_utils import printd

#? test that omit dyads and the entire process work in this way:
# omit the edges of one node and see if it's value has changed



#? test omittion no duplocacy 
# turn to dense adj and see if any of the edges is 2.
# def test_omittion_no_dup()
def test_no_duplicacy(edge_index, verbose=False):
    edge_idx_cpu = edge_index.clone().cpu()
    adj = to_dense_adj(edge_idx_cpu)[0]
    if torch.max(adj) > 1:
        if verbose:
            printd(f'there are duplicate edges in the graph at indices {torch.where(adj > 1)[0]}')
        return False
    return True


def calc_grad_no_tricks(x, edge_array, non_edge_array, lorenz):
    '''calculates the gradient using a loop. if there is edge omittion it should be done before calling this function.'''
    
    B = torch.ones(x.shape[1])
    if lorenz:
        B[x.shape[1]//2:] = -1
    grads = torch.zeros(x.shape)
    for i in range(len(x)):
        feats = x[i]
        edges_with_x = edge_array[:, torch.logical_or(edge_array[0] == i, edge_array[1] == i)]
        non_edges_with_x = non_edge_array[:,torch.logical_or(non_edge_array[0] == i, non_edge_array[1] == i)]
        grad = 0
        for edge in edges_with_x.T:
            feats_other = x[edge[edge != i]]
            inner_prod = feats @ (B*feats_other).T
            grad += B*feats_other*1/(torch.exp(inner_prod) - 1)
        for non_edge in non_edges_with_x.T:
            feats_other = x[non_edge[non_edge != i]]
            grad += -B*feats_other
        grads[i] = grad
    
    return grads

def test_omit_dyads_trainer_and_no_tricks(verbose=False, ds_name='smallBipart'):
    trainer_clam = Trainer(dataset_name=ds_name, model_name='ieclam', device='cpu')
    param_dict = {'dim_feat': 2,  
                    's_reg': 0.0,
                    'l1_reg': 0.0}
    trainer_clam.create_clamiter(param_dict)
    trainer_clam.data.x = trainer_clam.clamiter.init_node_feats(num_nodes=trainer_clam.data.num_nodes, init_type='small_gaus')


    # DEFINE THE EDGES TO OMIT
    edge_index = trainer_clam.data.edge_index
    num_nodes = trainer_clam.data.num_nodes
    num_edges = edge_index.shape[1]
    num_neg_samples = num_nodes*num_nodes - num_nodes - num_edges
    dyads_to_omit = utils.get_dyads_to_omit(edge_index, p_sample_edge=0.3)
    # ==================================

    #calculate with DIRECT SUMMATION and no trick
    non_edge_index = sort_edge_index(negative_sampling(trainer_clam.data.edge_index, num_neg_samples=num_neg_samples))
    mask_edges_omitted = edge_mask_of_selected_edges(edge_index, dyads_to_omit[0])
    mask_non_edges_omitted = edge_mask_of_selected_edges(non_edge_index, dyads_to_omit[1])

    edge_index_filtered = edge_index[:, ~mask_edges_omitted]
    non_edge_index_filtered = non_edge_index[:, ~mask_non_edges_omitted]
    grad_no_tricks = calc_grad_no_tricks(trainer_clam.data.x, edge_index_filtered, non_edge_index_filtered, lorenz=True)
    #======================================================

    # SUMMATION TRICK in clamiter forward function
    trainer_clam.data.edge_index, trainer_clam.data.edge_attr = trainer_clam.omit_dyads(dyads_to_omit)
    node_mask = torch.ones(trainer_clam.data.x.shape[0], dtype=torch.bool)

    grad_trainer = trainer_clam.clamiter.forward(trainer_clam.data, node_mask=node_mask)
    # ======================================================
    
    #? TEST direct summation == summation trick
    assert torch.allclose(grad_no_tricks, 2*grad_trainer, atol=1e-4), f'''
the no tricks gradient is not the same as the gradient in trainer->clamiter->forward
grad_no_tricks: {grad_no_tricks}
2*grad_trainer: {2*grad_trainer}
grad_trainer: {grad_trainer}
    '''
    #? ==================================================
    if verbose:
        printd('omitted the dyads and compared the gradient calculation with direct sum and the summation trick')
        #! notice that the edge_index is changed by the omit edges function!
        plot_adj(to_dense_adj(trainer_clam.data.edge_index_original)[0],dyads_to_omit)
    return True

def test_densify_test_from_train(verbose=False):
    '''in this test we make a simple dataset with train and test nodes and see that the correct edges are added and non edges that need not be added are not added.'''
    num_nodes = 8
    train_node_mask = torch.tensor([True, False, True, True, False, False, True, False])
    test_node_mask = ~train_node_mask
    edge_index = to_undirected(torch.tensor([[0, 0, 0, 1, 1, 2, 3, 3, 4],
                                             [1, 2, 4, 5, 7, 6, 4, 7, 7]]))
    edge_attr = torch.ones(edge_index.shape[1])
    ds = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, train_node_mask=train_node_mask)
    fat_ds = densify_ds_via_train_nodes(ds, train_node_mask, test_node_mask)

    edge_index_to_compare = sort_edge_index(
        to_undirected(torch.tensor(
            [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 4, 3],
             [1, 2, 4, 6, 2, 4, 5, 7, 4, 6, 4, 7, 7]])))
    
    return (fat_ds.edge_index == edge_index_to_compare).all()

def test_get_fat_ds_test(verbose=False):
    '''in this test we make a simple dataset with train and test nodes and see that the correct edges are added and non edges that need not be added are not added.'''
    num_nodes = 8
    train_node_mask = torch.tensor([True, False, True, True, False, False, True, False])
    test_node_mask = ~train_node_mask
    edge_index = to_undirected(torch.tensor([[0, 0, 0, 1, 1, 2, 3],
                                             [1, 2, 4, 5, 7, 6, 4]]))
    edge_attr = torch.ones(edge_index.shape[1])
    ds = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, train_node_mask=train_node_mask)
    fat_ds = get_fat_ds_test_OLD(ds, train_node_mask, test_node_mask)

    edge_index_to_compare = sort_edge_index(
        to_undirected(torch.tensor(
            [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 5],
             [1, 2, 4, 5, 6, 7, 2, 4, 5, 7, 4, 6, 4, 7]])))
    #! if this fails some time, know that it's because i added 1,2 and 2,4 as an edge (which is supposed to happen and weren't there for some reason....)
    return (fat_ds.edge_index == edge_index_to_compare).all()
  

def check_same_device(data, verbose=False):
    device = data.edge_index.device
    diff_device_list = []
    tbr = True
    for key, value in data:
        if hasattr(value, 'device') and value.device != device:
            tbr = False
            diff_device_list.append(key)
    if verbose:
        printd(f"\ndevice of edge_index: {device}")
        if not tbr:
            printd(f"\nthe following attributes are not on the same device as the edge_index:\n {diff_device_list}")
    
    return tbr