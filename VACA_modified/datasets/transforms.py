import torch


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.tensor(sample)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToOneHot:

    def __init__(self, n_dims=10):
        self.n_dims = n_dims

    def __call__(self, target):
        y_onehot = torch.FloatTensor(self.n_dims)
        y_onehot.zero_()
        y_onehot[target] = 1
        return y_onehot

    def __repr__(self):
        return self.__class__.__name__ + '()'
