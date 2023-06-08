import torch.nn as nn

from modules.blocks.pna import PNAConv
from utils.activations import get_activation
from utils.constants import Cte

"""
Code build from https://github.com/lukecavabarrett/pna
"""


class PNAModule(nn.Module):
    """
    Principal Neighborhood aggregation (PNA)

    Output activation is Identity
    """

    def __init__(self, c_list, deg,
                 edge_dim=None,
                 drop_rate=0.1,
                 act_name=Cte.RELU,
                 aggregators=None,
                 scalers=None,
                 residual=False):
        """

        Args:
            c_list:
            deg: In-degree histogram over training data
            edge_dim:
            drop_rate:
            act_name:
            aggregators:
            scalers:
            residual:
        """

        super().__init__()
        if aggregators is None:
            aggregators = ['sum', 'min', 'max', 'std']
        if scalers is None:
            scalers = ['identity', 'amplification', 'attenuation']

        self.convs = nn.ModuleList()
        self.activs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        if residual:
            self.residuals = nn.ModuleList()
        else:
            self.residuals = None

        self.residual = residual

        for i, (in_ch, out_ch) in enumerate(zip(c_list[:-1], c_list[1:])):
            conv = PNAConv(in_channels=in_ch, out_channels=out_ch, aggregators=aggregators,
                           edge_dim=edge_dim,
                           scalers=scalers, deg=deg, post_layers=1)

            self.convs.append(conv)
            act = get_activation(act_name if (i + 1) < len(c_list[:-1]) else Cte.IDENTITY)
            self.activs.append(act)

            dropout = nn.Dropout(drop_rate)
            self.dropouts.append(dropout)
            if self.residual:
                fc = nn.Linear(in_features=in_ch, out_features=out_ch)
                self.residuals.append(fc)

        self.drop_rate = drop_rate
        self.act_name = act_name

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        """

        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            edge_attr:
            **kwargs:

        Returns:

        """

        for i, (conv, act, dout) in enumerate(zip(self.convs, self.activs, self.dropouts)):
            h = act(conv(x, edge_index, edge_attr))
            if self.residual:
                x = h + self.residuals[i](x)
                x = dout(x)
            else:
                x = dout(h)
        return x
