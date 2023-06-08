from typing import List, Any

import numpy as np
import torch
import torch.nn as nn

import models._density_estimators as estims
from utils.activations import get_activation
from utils.constants import Cte
from utils.likelihoods import get_likelihood
from utils.probabilistic_model import ProbabilisticModelSCM


class HVACAModule(nn.Module):
    """
    Heterogeneous VACA Module
    """

    def __init__(self, likelihoods_x: List[List[Any]],
                 h_dim_list_dec: List[int],
                 h_dim_list_enc: List[int],
                 z_dim: int,
                 m_layers: int,  # Number of layers for the message MLP of the decoder
                 deg: List[float] = None,
                 edge_dim: int = None,
                 residual: int = 0,  # Use resitual network in message passing
                 drop_rate: float = 0.0,
                 act_name: str = Cte.RELU,
                 distr_z: str = 'normal',
                 architecture: str = None,
                 norm_categorical: bool = False,
                 norm_by_dim: int = 0,
                 K: int = 1):
        super(HVACAModule, self).__init__()

        self.K = K

        likelihood_z = get_likelihood(distr_z, z_dim)
        num_nodes = len(likelihoods_x)

        prob_model_x = ProbabilisticModelSCM(likelihoods=likelihoods_x,
                                             embedding_size=h_dim_list_dec[-1],
                                             act_name=act_name,
                                             drop_rate=drop_rate,
                                             norm_categorical=norm_categorical,
                                             norm_by_dim=norm_by_dim)

        # Instantiate Encoder embedding

        dim_input_encoder = h_dim_list_enc[0]

        self._encoder_embeddings = nn.ModuleList()
        for lik_i in likelihoods_x:
            x_dim_i = np.sum([lik_ij.domain_size for lik_ij in lik_i])
            if x_dim_i > 2 * dim_input_encoder:
                embed_i = nn.Sequential(nn.Linear(x_dim_i, 2 * dim_input_encoder, bias=True),
                                        get_activation(act_name),
                                        nn.Dropout(drop_rate),
                                        nn.Linear(2 * dim_input_encoder, dim_input_encoder, bias=True),
                                        get_activation(act_name),
                                        nn.Dropout(drop_rate))
            else:
                embed_i = nn.Sequential(nn.Linear(x_dim_i, dim_input_encoder, bias=True),
                                        get_activation(act_name),
                                        nn.Dropout(drop_rate))
            self._encoder_embeddings.append(embed_i)

        self.dim_input_enc = h_dim_list_enc[0]
        c_list = []
        c_list.extend(h_dim_list_enc)
        c_list.append(likelihood_z.params_size)
        # Instantiate Encoder module
        if architecture == 'pna':
            from modules.pna import PNAModule

            self.encoder_module = PNAModule(c_list=c_list,
                                            deg=deg,
                                            edge_dim=edge_dim,
                                            drop_rate=drop_rate,
                                            act_name=act_name,
                                            aggregators=None,
                                            scalers=None,
                                            residual=residual)
        elif architecture == 'dgnn':  # Disjoint GNN
            from modules.disjoint_gnn import DisjointGNN
            self.encoder_module = DisjointGNN(c_list=c_list,
                                              m_layers=len(c_list) - 1,  # We can only have 1 message passing step
                                              edge_dim=edge_dim,
                                              num_nodes=num_nodes,
                                              drop_rate=drop_rate,
                                              residual=residual,
                                              act_name=act_name,
                                              aggr='add')

        elif architecture == 'dpna':  # Disjoint PNA
            from modules.disjoint_pna import DisjointPNA
            self.encoder_module = DisjointPNA(c_list=c_list,
                                              m_layers=len(c_list) - 1,  # We can only have 1 message passing step
                                              edge_dim=edge_dim,
                                              deg=deg,
                                              num_nodes=num_nodes,
                                              aggregators=None,
                                              scalers=None,
                                              drop_rate=drop_rate,
                                              act_name=act_name,
                                              residual=residual)
        else:
            raise NotImplementedError

        c_list = [z_dim]
        c_list.extend(h_dim_list_dec)
        # Instantiate Decoder module
        if architecture == 'pna':
            from modules.pna import PNAModule
            self.decoder_module = PNAModule(c_list=c_list,
                                            deg=deg,
                                            edge_dim=edge_dim,
                                            drop_rate=drop_rate,
                                            act_name=act_name,
                                            aggregators=None,
                                            scalers=None,
                                            residual=residual)
        elif architecture == 'dgnn':
            from modules.disjoint_gnn import DisjointGNN

            self.decoder_module = DisjointGNN(c_list=c_list,
                                              m_layers=m_layers,
                                              edge_dim=edge_dim,
                                              num_nodes=num_nodes,
                                              drop_rate=drop_rate,
                                              residual=residual,
                                              act_name=act_name,
                                              aggr='add')

        elif architecture == 'dpna':
            from modules.disjoint_pna import DisjointPNA
            self.decoder_module = DisjointPNA(c_list=c_list,
                                              m_layers=m_layers,  # We can only have 1 message passing step
                                              edge_dim=edge_dim,
                                              deg=deg,
                                              num_nodes=num_nodes,
                                              aggregators=None,
                                              scalers=None,
                                              drop_rate=drop_rate,
                                              act_name=act_name,
                                              residual=residual)
        else:
            raise NotImplementedError

        # Instantiate Decoder embedding

        self.z_dim = z_dim

        self.num_nodes = len(likelihoods_x)

        node_dim_max = max(prob_model_x.node_dim_list)

        self.x0_size = self.num_nodes * node_dim_max

        self.node_dim_max = node_dim_max

        self.likelihood_z = likelihood_z
        self.prob_model_x = prob_model_x

        self.distr_z = distr_z

    def encoder_params(self):
        params = list(self.encoder_module.parameters()) + list(self._encoder_embeddings.parameters())
        return params

    def decoder_params(self):
        params = list(self.decoder_module.parameters()) + list(self.prob_model_x.parameters())
        return params

    def set_z_prior_distr(self, device):
        if self.distr_z == Cte.CONTINOUS_BERN:  # Continous Bernoulli
            self.z_prior_distr = torch.distributions.ContinuousBernoulli(
                probs=0.5 * torch.ones(self.hparams.latent_dim).to(device))
        elif self.distr_z == Cte.EXPONENTIAL:  # Exponential
            self.z_prior_distr = torch.distributions.Exponential(
                rate=0.2 * torch.ones(self.hparams.latent_dim).to(device))
        elif self.distr_z == Cte.BETA:  # Beta
            self.z_prior_distr = torch.distributions.Beta(
                concentration0=torch.ones(self.hparams.latent_dim).to(device),
                concentration1=torch.ones(self.hparams.latent_dim).to(device))
        elif self.distr_z == Cte.GAUSSIAN:
            self.z_prior_distr = torch.distributions.Normal(torch.zeros(self.z_dim).to(device),
                                                            torch.ones(self.z_dim).to(device))
        else:
            raise NotImplementedError

    def encoder_embeddings(self, X):

        X_0 = X.view(-1, self.x0_size)

        embeddings = []
        for i, embed_i in enumerate(self._encoder_embeddings):
            X_0_i = X_0[:, (i * self.node_dim_max):((i + 1) * self.node_dim_max)]
            H_i = embed_i(X_0_i[:, :self.prob_model_x.node_dim_list[i]])
            embeddings.append(H_i)

        return torch.cat(embeddings, dim=1).view(-1, self.dim_input_enc)

    def encoder(self, X, edge_index, edge_attr=None, return_mean=False, **kwargs):
        logits = self.encoder_module(self.encoder_embeddings(X),
                                     edge_index,
                                     edge_attr=edge_attr, **kwargs)
        if return_mean:
            mean, qz_x = self.likelihood_z(logits, return_mean=True)
            return mean, qz_x
        else:
            qz_x = self.likelihood_z(logits)
            return qz_x

    def decoder(self, Z, edge_index, edge_attr=None, return_type=None, **kwargs):
        logits = self.decoder_module(Z, edge_index, edge_attr, **kwargs)

        if return_type == 'mean':
            mean, px_z = self.prob_model_x(logits, return_mean=True)
            return mean, px_z
        elif return_type == 'sample':
            mean, px_z = self.prob_model_x(logits, return_mean=True)
            return px_z.sample(), px_z
        else:
            px_z = self.prob_model_x(logits)
            return px_z

    def compute_log_w(self, data, K, mask=None):
        """
        IWAE:  log(1\K \sum_k w_k) w_k = p(x, z_i)/ q(z_i | x)
            log_wi = log  p(x, z_i) - log q(z_i | x)
        Args:
            data:
            K:
            mask:

        Returns:

        """

        x = data.x.clone()

        assert mask is None

        log_w = []
        for k in range(K):
            qz_x = self.encoder(x, data.edge_index, edge_attr=data.edge_attr, node_ids=data.node_ids)
            z = qz_x.rsample()

            px_z_k = self.decoder(z, data.edge_index, edge_attr=data.edge_attr, node_ids=data.node_ids)

            log_prob_qz_x = qz_x.log_prob(z).view(data.num_graphs, -1).sum(-1)  # Summing over dim(z)*num_nodes
            log_prob_pz = self.z_prior_distr.log_prob(z).view(data.num_graphs, -1).sum(-1)

            log_prob_px_z = px_z_k.log_prob(self.get_x_graph(data, 'x')).sum(-1)

            log_w_k = log_prob_px_z + log_prob_pz - log_prob_qz_x

            log_w.append(log_w_k)

        log_w = torch.stack(log_w, dim=0)

        # [K, N]
        return log_w.T

    def get_x_graph(self, data, attr):
        x = getattr(data, attr)
        mask = data.mask.view(data.num_graphs, -1)[0]
        return x.view(data.num_graphs, -1)[:, mask]

    def forward(self, data, estimator, beta=1.0):

        x = data.x.clone()

        mask = None

        if estimator == 'elbo':

            qz_x = self.encoder(x,
                                data.edge_index,
                                edge_attr=data.edge_attr,
                                node_ids=data.node_ids)
            z = qz_x.rsample()

            px_z = self.decoder(z, data.edge_index, edge_attr=data.edge_attr, node_ids=data.node_ids)

            log_prob_x = px_z.log_prob(self.get_x_graph(data, 'x')).sum(1).mean()
            kl_z = torch.distributions.kl.kl_divergence(qz_x, self.z_prior_distr).view(data.num_graphs, -1).sum(
                1).mean()

            elbo = log_prob_x - beta * kl_z

            data = {'log_prob_x': log_prob_x,
                    'kl_z': kl_z}

            return elbo, data
        elif estimator == 'iwae':
            log_w = self.compute_log_w(data=data, K=self.K, mask=mask)
            objective, _ = estims.IWAE(log_w, trick=True)
            return objective.mean(), {}

        elif estimator == 'iwaedreg':
            log_w, zs = self.compute_log_w_dreg(data=data, K=self.K)
            objective, _ = estims.IWAE_dreg(log_w, zs)
            return objective.mean(), {}

        else:
            raise NotImplementedError

    @torch.no_grad()
    def reconstruct(self, data, use_mean_encoder=True):
        z_mean, qz_x = self.encoder(data.x, data.edge_index, edge_attr=data.edge_attr,
                                    return_mean=True, node_ids=data.node_ids)

        z = z_mean if use_mean_encoder else qz_x.rsample()
        x_hat, _ = self.decoder(z, data.edge_index, edge_attr=data.edge_attr,
                                return_type='mean', node_ids=data.node_ids)

        # Shape of x_hat: [num_graphs, total_dim_nodes]
        return z_mean, x_hat
