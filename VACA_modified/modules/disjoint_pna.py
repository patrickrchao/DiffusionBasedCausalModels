from typing import List

import torch.nn as nn
from torch import Tensor

from modules.blocks.disjoint_dense import DisjointDense
from modules.blocks.pna.disjoint_pna import DisjointPNAConv
from utils.activations import get_activation
from utils.constants import Cte

"""
Code build from https://github.com/lukecavabarrett/pna
"""


class DisjointPNA(nn.Module):
    """
    Disjoint parameters for each edge connection
    """

    def __init__(self, c_list: List[int],  # [1,2,2]
                 m_layers: int,  # Number of layers per message
                 edge_dim: int,
                 deg: Tensor,
                 num_nodes: int = None,
                 aggregators: List[str] = None,
                 scalers: List[str] = None,
                 drop_rate: float = 0.1,
                 residual: int = 0,
                 act_name: str = Cte.RELU,
                 ):

        super().__init__()
        assert (len(c_list) - 1) % m_layers == 0

        if aggregators is None:
            aggregators = ['sum', 'min', 'max', 'std']
        if scalers is None:
            scalers = ['identity']

        #
        self.residual = residual
        self.convs = nn.ModuleList()
        self.activs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        if residual:
            assert num_nodes is not None
            self.residuals = nn.ModuleList()
        else:
            self.residuals = None
        self.num_steps_mp = (len(c_list) - 1) // m_layers  # Number of steps of message passing

        for i in range(self.num_steps_mp):
            m_channels = c_list[(m_layers * i):(m_layers * (i + 1) + 1)]
            net = DisjointPNAConv(m_channels=m_channels.copy(),
                                  edge_dim=edge_dim,
                                  num_nodes=num_nodes,
                                  aggregators=aggregators,
                                  scalers=scalers,
                                  deg=deg,
                                  act_name=act_name,
                                  drop_rate=drop_rate)

            self.convs.append(net)
            act = get_activation(act_name if (i + 1) < self.num_steps_mp else Cte.IDENTITY)
            self.activs.append(act)

            dropout = nn.Dropout(drop_rate if (i + 1) < self.num_steps_mp else 0.0)
            self.dropouts.append(dropout)
            if self.residual:
                fc = DisjointDense(in_dimension=m_channels[0], out_dimension=m_channels[-1], num_disjoint=num_nodes)
                self.residuals.append(fc)
        self.drop_rate = drop_rate

        self.act_name = act_name

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        """

        Args:
            x: Input features per node
            edge_index:  List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            edge_attr:
            **kwargs:

        Returns:

        """
        node_ids = kwargs['node_ids']
        for i, (conv, act, dout) in enumerate(zip(self.convs, self.activs, self.dropouts)):
            h = act(conv(x, edge_index, edge_attr, node_ids))
            if self.residual:

                x = h + self.residuals[i](x, node_ids)
                x = dout(x)
            else:
                x = dout(h)

        return x
