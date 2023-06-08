import torch
from sklearn import preprocessing

flatten = lambda t: [item for sublist in t for item in sublist]


class MaskedTensorStandardScaler:
    def __init__(self, list_dim_to_scale_x0, list_dim_to_scale, total_num_dimensions):
        self.list_dim_to_scale_x0 = list_dim_to_scale_x0  # [0, 1, 4, 5 ,7 ,8]
        self.list_dim_to_scale = list_dim_to_scale  # [0, 1, 4, 5 ,7 ,8]
        self.total_num_dimensions = total_num_dimensions
        self.scaler = preprocessing.StandardScaler()

    def fit(self, x):
        if x.shape[1] != self.total_num_dimensions:
            self.scaler.fit(x[:, self.list_dim_to_scale_x0])
        else:
            self.scaler.fit(x[:, self.list_dim_to_scale])

    def transform(self, x):
        if x.shape[1] != self.total_num_dimensions:
            x_scaled = self.scaler.transform(x[:, self.list_dim_to_scale_x0])
            x[:, self.list_dim_to_scale_x0] = x_scaled
        else:
            x_scaled = self.scaler.transform(x[:, self.list_dim_to_scale])
            x[:, self.list_dim_to_scale] = x_scaled
        return torch.tensor(x)

    def inverse_transform(self, x):
        if x.shape[1] != self.total_num_dimensions:
            x_unscaled = self.scaler.inverse_transform(x[:, self.list_dim_to_scale_x0])
            x[:, self.list_dim_to_scale_x0] = torch.tensor(x_unscaled)
        else:
            x_unscaled = self.scaler.inverse_transform(x[:, self.list_dim_to_scale])
            x[:, self.list_dim_to_scale] = torch.tensor(x_unscaled)
        return x


class MaskedTensorLikelihoodScaler:
    def __init__(self, likelihoods, mask_x0):
        self.likelihoods = flatten(likelihoods)
        self.mask_x0 = mask_x0

        self.total_num_dim_x0 = len(mask_x0)

        self.dim_list = []
        for lik in self.likelihoods:
            self.dim_list.append(lik.domain_size)

    def fit(self, x):
        x = torch.tensor(x).type(torch.float32)
        x = x[:, self.mask_x0] if self.total_num_dim_x0 == x.shape[1] else x
        x_list = torch.split(x, split_size_or_sections=self.dim_list, dim=1)

        for lik_i, x_i in zip(self.likelihoods, x_list):
            lik_i.fit(x_i)

    def transform(self, x):
        x = torch.tensor(x).type(torch.float32)

        if self.total_num_dim_x0 == x.shape[1]:
            x_tmp = x[:, self.mask_x0]
        else:
            x_tmp = x
        x_list = torch.split(x_tmp, split_size_or_sections=self.dim_list, dim=1)
        x_norm = []
        for lik_i, x_i in zip(self.likelihoods, x_list):
            x_norm.append(lik_i.normalize_data(x_i))
        x_norm = torch.cat(x_norm, dim=1)

        if self.total_num_dim_x0 == x.shape[1]:
            x[:, self.mask_x0] = x_norm
            return x
        else:
            return x_norm

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x_norm):
        x_norm = torch.tensor(x_norm)
        #Added by Patrick
        x_norm = x_norm.cpu()
        if self.total_num_dim_x0 == x_norm.shape[1]:
            x_tmp = x_norm[:, self.mask_x0]
        else:
            x_tmp = x_norm

        x_list = torch.split(x_tmp, split_size_or_sections=self.dim_list, dim=1)
        x = []
        for lik_i, x_i in zip(self.likelihoods, x_list):
            x.append(lik_i.denormalize_data(x_i))
        x = torch.cat(x, dim=1)
        if self.total_num_dim_x0 == x.shape[1]:
            x_norm[:, self.mask_x0] = x
            return x_norm
        else:
            return x


class TensorScaler:
    def __init__(self, scaler):
        self.scaler = scaler

    def transform(self, x):
        return torch.tensor(self.scaler.transform(x))

    def inverse_transform(self, x):
        return torch.tensor(self.scaler.inverse_transform(x))


