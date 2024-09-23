import os
import scipy.sparse as sp
import scipy.io
import torch

from torch_geometric import EdgeIndex
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import coalesce, remove_self_loops
from torch_geometric import utils

from utils.printing_utils import printd

#* pyg supplement functions. 



def load_data(data_source):
    data = scipy.io.loadmat("gae/data/{}.mat".format(data_source))
    # labels = data["gnd"]
    return data

def sparse_matrix_to_edge_index(sparse_matrix):
    
    # Ensure the matrix is in COO format
    sparse_matrix = sparse_matrix.tocoo()

    # Extract row and column indices
    row = torch.tensor(sparse_matrix.row, dtype=torch.long)
    col = torch.tensor(sparse_matrix.col, dtype=torch.long)

    # Stack indices to form edge_index
    edge_index = torch.stack([row, col], dim=0)

    return edge_index

def has_repeating_edges(edge_index):
    
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    # Use a set to track seen edges
    seen_edges = set()

    # Check for duplicates
    for edge in edges:
        if edge in seen_edges:
            return True
        seen_edges.add(edge)

    return False

def keep_bidirectional_edges(edge_index):
    """
    Throw out directed edges
    
    Parameters:
    edge_index (torch.Tensor): The edge_index tensor of shape [2, num_edges].

    Returns:
    torch.Tensor: The edge_index tensor with only bidirectional edges.
    """
    # Convert edge_index to a list of tuples
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    # Use a set to track seen edges
    seen_edges = set()
    bidirectional_edges = []

    for edge in edges:
        if (edge[1], edge[0]) in seen_edges:
            bidirectional_edges.append(edge)
            bidirectional_edges.append((edge[1], edge[0]))
        seen_edges.add(edge)

    # Convert back to tensor
    bidirectional_edges = torch.tensor(bidirectional_edges, dtype=torch.long).t()

    return bidirectional_edges

def remove_duplicate_edges(edge_index):
    """
    Remove duplicate edges from the edge_index tensor.

    Parameters:
    edge_index (torch.Tensor): The edge_index tensor of shape [2, num_edges].

    Returns:
    torch.Tensor: The edge_index tensor with duplicate edges removed.
    """
    # Convert edge_index to a list of tuples
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    # Use a set to track unique edges
    unique_edges = list(set(edges))

    # Convert back to tensor
    unique_edges = torch.tensor(unique_edges, dtype=torch.long).t()

    return unique_edges


def get_undirected_edges(edge_index):
    """
    Identify and return undirected edges from the edge_index tensor.

    Parameters:
    edge_index (torch.Tensor): The edge_index tensor of shape [2, num_edges].

    Returns:
    torch.Tensor: The edge_index tensor with only undirected edges.
    """
    # Convert edge_index to a list of tuples
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    # Use a set to track seen edges
    seen_edges = set()
    undirected_edges = []

    for edge in edges:
        if (edge[1], edge[0]) in seen_edges:
            undirected_edges.append(edge)
            undirected_edges.append((edge[1], edge[0]))
        seen_edges.add(edge)

    # Convert back to tensor
    undirected_edges = torch.tensor(undirected_edges, dtype=torch.long).t()

    return undirected_edges

def get_directed_edges(edge_index):
    """
    Identify and return directed edges from the edge_index tensor.

    Parameters:
    edge_index (torch.Tensor): The edge_index tensor of shape [2, num_edges].

    Returns:
    torch.Tensor: The edge_index tensor with only directed edges.
    """
    # Convert edge_index to a list of tuples
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    # Use a set to track seen edges
    seen_edges = set(edges)
    directed_edges = []

    for edge in edges:
        if (edge[1], edge[0]) not in seen_edges:
            directed_edges.append(edge)

    # Convert back to tensor
    directed_edges = torch.tensor(directed_edges, dtype=torch.long).t()

    return directed_edges

def check_indices_in_edge_index(indices, edge_index):
    """
    Check if the edge_index array contains the indices in the list.

    Parameters:
    indices (list of tuples): List of index tuples to check.
    edge_index (torch.Tensor): The edge_index tensor of shape [2, num_edges].

    Returns:
    list of bool: Boolean array indicating the presence of each index in the edge_index.
    """
    # Convert edge_index to a set of tuples for efficient lookup
    edges_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    # Check if each index is in the edges_set
    result = [(i, j) in edges_set for i, j in indices]

    return result

def check_anomalies_in_edge_index(edge_index, anomaly_indices):
    """
    Check if the edge_index contains indices from the anomaly_indices array.

    Parameters:
    edge_index (torch.Tensor): The edge_index tensor of shape [2, num_edges].
    anomaly_indices (torch.Tensor): The tensor containing anomaly indices.

    Returns:
    torch.Tensor: Boolean tensor indicating the presence of anomaly indices in the edge_index.
    """
    # Convert anomaly_indices to a set for efficient lookup
    anomaly_set = set(anomaly_indices.tolist())

    # Check if each edge contains an anomaly index
    result = [(i in anomaly_set and j in anomaly_set) for i, j in zip(edge_index[0].tolist(), edge_index[1].tolist())]

    # Convert result to a tensor
    result_tensor = torch.tensor(result, dtype=torch.bool)

    return result_tensor

def compare_datas(data1, data2):
    att1_notin2 = set()
    att2_notin1 = set()
    diffs = set()

    # Get the attributes of the datasets
    attrs1 = data1.__dict__['_store']
    attrs2 = data2.__dict__['_store']
    
    keys1 = set(attrs1.keys())
    keys2 = set(attrs2.keys())
    # first compare if they have the same attributes
    for key in keys1:
        if key not in keys2:
            att1_notin2.add(key)
        else:
            try:
                if isinstance(attrs1[key], torch.Tensor) and isinstance(attrs2[key], torch.Tensor): 
                    if not torch.equal(attrs1[key], attrs2[key]):
                        diffs.add(key)
                else:
                    if (torch.tensor(attrs1[key]) != torch.tensor(attrs2[key])).any():
                        diffs.add(key)
            except Exception as e:
                printd(f'Error when comparing data objects: {e}')

    for key in keys2:
        if key not in keys1:
            att2_notin1.add(key)
    
    return att1_notin2, att2_notin1, diffs


def check_single_device(data, verbose=False):
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



def dropout_edge_undirected(edge_index, p=0.5):
    
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    row, col = edge_index
    edge_index_directed = edge_index[:, row < col]

    row_directed, col_directed = edge_index_directed

    edge_mask_directed = torch.rand(row_directed.size(0), device=edge_index.device) >= p
    edge_index_directed_retained = edge_index_directed[:, edge_mask_directed]

    edge_index_retained = torch.cat([edge_index_directed_retained, edge_index_directed_retained.flip(0)], dim=1)
    edge_mask_retained = torch.cat([edge_mask_directed, edge_mask_directed])
    edge_index_orig_rearange = torch.cat([edge_index_directed, edge_index_directed.flip(0)], dim=1)

    return edge_index_retained, edge_mask_retained, edge_index_orig_rearange

# i just want an edge_attr mask for the edges that were dropped

def edge_mask_drop_and_rearange(edge_index, p):
    '''
    drops edges with probability p.
    returns the edge index rearaged and the mask for the dropped edges
    it's important to get the rearanged edge_index to get the correct mask because it's hard to get it for undirected'''
    row, col = edge_index
    edge_index_directed = edge_index[:, row < col]

    row_directed, col_directed = edge_index_directed
    
    edge_mask_directed_retain = torch.rand(row_directed.size(0), device=edge_index.device) >= p
    
    edge_mask_retain = torch.cat([edge_mask_directed_retain, edge_mask_directed_retain])
    edge_index_orig_rearange = torch.cat([edge_index_directed, edge_index_directed.flip(0)], dim=1)
    #! very important to use the new edge index otherwise the positions of the dropped edges is not correct!
    return edge_index_orig_rearange, edge_mask_retain



# @functional_transform('my_two_hop')
# class MyTwoHop(BaseTransform):
#     r"""Adds the two hop edges to the edge indices
#     (functional name: :obj:`two_hop`).
#     """
#     def forward(self, data: Data) -> Data:
#         assert data.edge_index is not None
#         edge_index, edge_attr = data.edge_index, data.edge_attr
#         N = data.num_nodes

#         edge_index = EdgeIndex(edge_index, sparse_size=(N, N))
#         edge_index = edge_index.sort_by('row')[0]
#         edge_index2 = edge_index.matmul(edge_index)[0].as_tensor()
#         edge_index2, _ = remove_self_loops(edge_index2)
#         edge_index = torch.cat([edge_index, edge_index2], dim=1)

#         if edge_attr is not None:
#             # We treat newly added edge features as "zero-features":
#             edge_attr2 = edge_attr.new_zeros(edge_index2.size(1), *edge_attr.size()[1:]) + 1
#             edge_attr = torch.cat([edge_attr, edge_attr2], dim=0)

#         data.edge_index, data.edge_attr = coalesce(edge_index, edge_attr, N)

#         return data

def my_two_hop(data):
    '''densify the edges with with attr 1. if one of the edges with attr 0 is produced, set it's attr to 1'''
    assert data.edge_index is not None
    edge_index, edge_attr = data.edge_index, data.edge_attr
    N = data.num_nodes

    # Sort edge_index by the attribute values
    # sorted_indices = edge_attr.argsort(dim=0, descending=True)
    # edge_index = edge_index[:, sorted_indices]
    # edge_attr = edge_attr[sorted_indices]
    
    #densify the edge with attribute 1
    edges_to_densify = edge_index[:, edge_attr]

    edges_to_densify = EdgeIndex(edges_to_densify, sparse_size=(N, N))
    edges_to_densify = edges_to_densify.sort_by('row')[0]
    # all of the 2hop edges V:
    edges_densified = edges_to_densify.matmul(edges_to_densify)[0].as_tensor()
    edges_densified, _ = remove_self_loops(edges_densified)
    edge_index = torch.cat([edge_index, edges_densified], dim=1)

    # We treat newly added edge features as "zero-features":
    attr_densified = torch.ones(edges_densified.size(1)).bool()
    edge_attr = torch.cat([edge_attr, attr_densified], dim=0)

    edge_index, edge_attr = coalesce(edge_index, edge_attr, N, reduce="max")

    return edge_index, edge_attr

