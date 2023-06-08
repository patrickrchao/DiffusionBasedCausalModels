
from torch_geometric.utils import dense_to_sparse
import torch

import networkx as nx
class Adjacency:
    def __init__(self, adj):

        self.adj_matrix = adj

        self.num_nodes = adj.shape[0]
        self.num_edges = int(adj.sum())

        self.edge_index, _ = dense_to_sparse(torch.tensor(self.adj_matrix))
        self.edge_attr = torch.eye(self.num_edges, self.num_edges)


        #  Adjacency intervention
        self.adj_matrix_i = None
        self.edge_attr_i = None
        self.edge_index_i = None


        return



    def set_diagonal(self):

        self.adj_matrix_i = torch.eye(self.num_nodes)

        self.edge_index_i, _ = dense_to_sparse(torch.tensor(self.adj_matrix_i))

        edge_attr_i = []
        for i in range(self.edge_index_i.shape[1]):
            for j in range(self.num_edges):
                if all(self.edge_index_i[:, i] == self.edge_index[:, j]):
                    edge_attr_i.append(self.edge_attr[j])
                    break


        self.edge_attr_i  = torch.stack(edge_attr_i, 0)





    def set_intervention(self, node_id_list, add_self_loop=True):

        self.adj_matrix_i = self.adj_matrix.copy()
        for node_id in node_id_list:
            self.adj_matrix_i[:, node_id] = 0.0
            if add_self_loop: self.adj_matrix_i[node_id, node_id] = 1.0

        self.edge_index_i, _ = dense_to_sparse(torch.tensor(self.adj_matrix_i))
        edge_attr_i = []
        for i in range(self.edge_index_i.shape[1]):
            for j in range(self.num_edges):
                if all(self.edge_index_i[:, i] == self.edge_index[:, j]):
                    edge_attr_i.append(self.edge_attr[j])
                    break

        self.edge_attr_i  = torch.stack(edge_attr_i, 0)

    def clean_intervention(self):
        #  Adjacency intervention
        self.adj_matrix_i = None
        self.edge_attr_i = None
        self.edge_index_i = None

