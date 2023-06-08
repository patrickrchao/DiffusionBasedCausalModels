import torch.nn as nn

from utils.activations import get_activation


def basic_dense_block(in_dim, out_dim, activation_name, drop_rate=0.0, bn=True, **kwargs):
    act = get_activation(activation_name)
    dense = nn.Linear(in_dim, out_dim)
    drop_layer = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
    bn_layer = nn.BatchNorm1d(out_dim) if bn else nn.Identity()
    return nn.Sequential(dense, bn_layer, act, drop_layer)
