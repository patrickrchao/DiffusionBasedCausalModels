import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric


def normalize_adj(adj, how='sym'):
    if how == 'sym':
        deg = adj.sum(1)
        deg_inv = 1 / np.sqrt(deg)
        deg_inv[deg_inv == float('inf')] = 0
        D_inv = np.diag(deg_inv)
        return D_inv @ adj @ D_inv
    elif how == 'row':  # Normalize by children
        deg = adj.sum(1)
        deg_inv = 1 / deg
        deg_inv[deg_inv == float('inf')] = 0
        D_inv = np.diag(deg_inv)
        return D_inv @ adj
    elif how == 'col':  # Normalize by parents
        deg = adj.sum(0)
        deg_inv = 1 / deg
        deg_inv[deg_inv == float('inf')] = 0
        D_inv = np.diag(deg_inv)
        return adj @ D_inv


def convert_nodes_to_df(G, node_id_col='node_id'):
    data = []
    for node_id, attr_dict in G.nodes(True):
        my_dict = {node_id_col: node_id}
        my_dict.update(attr_dict)
        data.append(my_dict)
    return pd.DataFrame(data)


def convert_edges_to_df(G):
    data = []
    for node_src, node_dst in G.edges:
        my_dict = {'node_src': node_src, 'node_dst': node_dst}
        my_dict.update(G.edges[(node_src, node_dst)])
        data.append(my_dict)
    return pd.DataFrame(data)


def from_networkx(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data
