import torch.nn as nn

from utils.constants import Cte


def get_activation(name):
    if name == Cte.TAHN:
        return nn.Tanh()
    elif name == Cte.RELU:
        return nn.ReLU()
    elif name == Cte.RELU6:
        return nn.ReLU6()
    elif name == Cte.SOFTPLUS:
        return nn.Softplus()
    elif name == Cte.RRELU:
        return nn.RReLU()
    elif name == Cte.LRELU:
        return nn.LeakyReLU(negative_slope=0.05)
    elif name == Cte.ELU:
        return nn.ELU()
    elif name == Cte.SELU:
        return nn.SELU()
    elif name == Cte.GLU:
        return nn.GLU()
    elif name == Cte.SIGMOID:
        return nn.Sigmoid()
    elif name == Cte.IDENTITY:
        return nn.Identity()
    else:
        raise NotImplementedError
