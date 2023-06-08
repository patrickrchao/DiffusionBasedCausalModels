from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj

from modules.blocks.disjoint_dense import DisjointDense
from utils.activations import get_activation
from utils.constants import Cte


class DisjointGConv(MessagePassing):
    """
    Disjoint Graph convolution
    """

    def __init__(self, m_channels: list,
                 edge_dim: int,
                 aggr: Optional[str] = 'add',
                 act_name: Optional[str] = 'relu',
                 drop_rate: Optional[float] = 0.0,
                 use_i_in_message_ij: Optional[bool] = False,
                 **kwargs):
        """

        Args:
            m_channels:
            edge_dim:
                one hot encoding of the index of the edge in the graph.
                I.e., edge_dim = # edges in our graph including self loops.
            aggr:
            act_name:
            drop_rate:
            use_i_in_message_ij:
            **kwargs:
        """

        super(DisjointGConv, self).__init__(aggr=aggr,
                                            node_dim=0, **kwargs)

        assert len(m_channels) >= 2
        self.m_net_list = nn.ModuleList()
        self.activs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.use_i_in_message_ij = use_i_in_message_ij
        self.m_layers = len(m_channels) - 1
        if self.use_i_in_message_ij: m_channels[0] = m_channels[0] * 2

        for i, (in_ch, out_ch) in enumerate(zip(m_channels[:-1], m_channels[1:])):
            m_net = DisjointDense(in_dimension=in_ch, out_dimension=out_ch, num_disjoint=edge_dim)
            self.m_net_list.append(m_net)
            act = get_activation(act_name if (i + 1) < len(m_channels[:-1]) else Cte.IDENTITY)
            self.activs.append(act)

            dropout = nn.Dropout(drop_rate if (i + 1) < len(m_channels[:-1]) else 0.0)
            self.dropouts.append(dropout)

        self.edge_dim = edge_dim

        self.reset_parameters()

    def reset_parameters(self):
        for m_net in self.m_net_list:
            m_net.reset_parameters()

    def forward(self, x: Tensor,
                edge_index: Adj,
                edge_attr: Tensor) -> Tensor:
        """

        Args:
            x:
            edge_index:
                edge_index = []
                edge_index.append([0,1])
                edge_index.append([2,2])
            edge_attr:

        Returns:

        """

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return out

    def message(self, x_i: Tensor,
                x_j: Tensor,
                edge_attr: Tensor) -> Tensor:
        """

        Args:
            x_i:
                are N nodes being updated
            x_j:
                is a neighbor of node x_i, could be itself if we have self-loops
            edge_attr:
                dimension self.edge_dim. In our case one-hot encoding

        Returns:

        """

        if self.use_i_in_message_ij:
            x = torch.cat([x_i, x_j], dim=1)
        else:
            x = x_j

        for i, (m_net, act, dout) in enumerate(zip(self.m_net_list, self.activs, self.dropouts)):
            h = act(m_net(x, one_hot_selector=edge_attr))
            x = dout(h)
        return x
