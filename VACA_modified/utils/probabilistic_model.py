from typing import List, Any, Dict

import torch
import torch.nn as nn

from utils.activations import get_activation

from utils.constants import Cte

flatten = lambda t: [item for sublist in t for item in sublist]


class HeterogeneousDistribution:
    def __init__(self, likelihoods: List[List[Any]],
                 norm_categorical: bool,  # True, False
                 norm_by_dim: bool  # Unsigned Integer
                 ):
        assert isinstance(likelihoods, list)

        self.norm_categorical = norm_categorical

        self.likelihoods = likelihoods
        self.params_size_list = []
        self.likelihood_name_list = []
        self.dim_list = []
        self.norm_by_dim = norm_by_dim
        for lik in likelihoods:
            self.params_size_list.append(lik.params_size)
            self.likelihood_name_list.append(lik.name)
            self.dim_list.append(lik.domain_size)

    def set_logits(self, logits):
        self.distributions = []
        logits_list = torch.split(logits, split_size_or_sections=self.params_size_list, dim=1)
        for lik_i, logits_i in zip(self.likelihoods, logits_list):
            self.distributions.append(lik_i(logits_i))

    @property
    def mean(self):
        means = []
        for i, distr in enumerate(self.distributions):
            if self.likelihood_name_list[i] in ['cat', 'ber', 'cb']:
                means.append(distr.probs)
            else:
                means.append(distr.mean)
        return torch.cat(means, dim=1)

    def sample(self, sample_shape=torch.Size()):
        samples = []
        for i, distr in enumerate(self.distributions):
            if self.likelihood_name_list[i] in [Cte.CATEGORICAL]:
                sample_i = distr.sample(sample_shape)
                y_onehot = torch.FloatTensor(distr.probs.shape)
                # In your for loop
                y_onehot.zero_()
                y_onehot.scatter_(1, sample_i.view(-1, 1), 1)
                sample_i = y_onehot
                samples.append(sample_i)
            else:
                samples.append(distr.sample(sample_shape))

        return torch.cat(samples, dim=1)

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError()

    def log_prob(self, value):
        '''
        [num_graphs, total_dim_nodes]
        '''
        value_list = torch.split(value, split_size_or_sections=self.dim_list, dim=1)

        log_probs = []
        for distr_name, value_i, distr_i in zip(self.likelihood_name_list, value_list, self.distributions):

            if distr_name in [Cte.CATEGORICAL]:
                num_categories = value_i.shape[1]
                value_i = torch.argmax(value_i, dim=-1)
                log_prob_i = distr_i.log_prob(value_i).view(-1, 1)
                if self.norm_categorical: log_prob_i = log_prob_i / num_categories
                log_probs.append(log_prob_i)
            else:
                # print(f"distr: {distr_i}")
                # print(f"max: {value_i.max()}")
                # print(f"min: {value_i.min()}")
                log_prob_i = distr_i.log_prob(value_i)
                if self.norm_by_dim == 1:  # Normalize by dimension
                    log_prob_i = log_prob_i / log_prob_i.shape[1]
                elif self.norm_by_dim > 1:
                    log_prob_i = log_prob_i / self.norm_by_dim
                log_probs.append(log_prob_i)

        return torch.cat(log_probs, dim=1)


class ProbabilisticModelSCM(nn.Module):
    def __init__(self, likelihoods: List[List[Any]],
                 embedding_size: int,
                 act_name: str,  # None
                 drop_rate: float,  # None
                 norm_categorical: bool,  # False
                 norm_by_dim: int  # False
                 ):

        """
        Args:
            likelihoods:
            embedding_size:
            act_name:
            drop_rate:
            norm_categorical:
            norm_by_dim:
        """

        '''


        '''
        super().__init__()

        flatten = lambda t: [item for sublist in t for item in sublist]

        self.num_nodes = len(likelihoods)
        self.total_x_dim = sum([lik.domain_size for lik in flatten(likelihoods)])

        self.norm_categorical = norm_categorical
        self.node_dim_list = []
        likelihood_node_params_size_list = []  # Size = num_nodes
        for lik_i in likelihoods:
            likelihood_node_i_params_size = 0
            dim_node_i = []
            for lik_ij in lik_i:
                likelihood_node_i_params_size += lik_ij.params_size
                dim_node_i.append(lik_ij.domain_size)
            self.node_dim_list.append(sum(dim_node_i))
            likelihood_node_params_size_list.append(likelihood_node_i_params_size)

        self.embedding_size = embedding_size
        self._decoder_embeddings = nn.ModuleList()
        for likelihood_node_params_size_i in likelihood_node_params_size_list:
            if likelihood_node_params_size_i > 2 * embedding_size:
                embed_i = nn.Sequential(nn.Linear(embedding_size, 2 * embedding_size, bias=True),
                                        get_activation(act_name),
                                        nn.Dropout(drop_rate),
                                        nn.Linear(2 * embedding_size, likelihood_node_params_size_i, bias=True))
            else:
                embed_i = nn.Linear(self.embedding_size, likelihood_node_params_size_i, bias=False)

            self._decoder_embeddings.append(embed_i)

        self.logits_dim = sum(likelihood_node_params_size_list)

        self.heterogeneous_distr = HeterogeneousDistribution(likelihoods=flatten(likelihoods),
                                                             norm_by_dim=norm_by_dim,
                                                             norm_categorical=norm_categorical)

    def forward(self, logits, return_mean=False):
        """

        Args:
            logits: [num_nodes, max_dim_node]
            return_mean:

        Returns:

        """
        d = logits.shape[1]
        logits_0 = logits.view(-1, self.num_nodes * d)  # Num graphs, max_dim_node*num_nodes

        logits_list = []
        for i, embed_i in enumerate(self._decoder_embeddings):
            logits_0i = logits_0[:, (i * d):((i + 1) * d)]
            logits_i = embed_i(logits_0i)
            logits_list.append(logits_i)

        logits = torch.cat(logits_list, dim=-1)
        assert logits.shape[1] == self.logits_dim

        p = self.heterogeneous_distr
        p.set_logits(logits)

        if return_mean:
            return p.mean, p
        else:
            return p
