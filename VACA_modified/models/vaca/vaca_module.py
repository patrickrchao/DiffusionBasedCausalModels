from typing import List, Any

import torch
import torch.nn as nn

import models._density_estimators as estims
from utils.constants import Cte
from utils.likelihoods import get_likelihood


class VACAModule(nn.Module):
    """
    VACA Module
    """

    def __init__(self, x_dim: int,
                 h_dim_list_dec: List[int],
                 h_dim_list_enc: List[int],
                 z_dim: int,
                 m_layers: int,  # Number of layers for the message MLP of the decode
                 deg: List[float] = None,
                 edge_dim: int = None,
                 num_nodes: int = None,
                 residual: int = 0,  # Use resitual network in message passing
                 drop_rate: float = 0.0,
                 act_name: str = Cte.RELU,
                 likelihood_x: Any = None,
                 distr_z: str = 'normal',
                 architecture: str = None,
                 K: int = 1):
        super(VACAModule, self).__init__()

        assert isinstance(x_dim, int)

        self.K = K

        likelihood_z = get_likelihood(distr_z, z_dim)

        c_list = [x_dim]
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
        c_list.append(likelihood_x.params_size)
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

        self.z_dim = z_dim

        self.likelihood_z = likelihood_z
        self.likelihood_x = likelihood_x

        self.distr_z = distr_z

    def encoder_params(self):
        return self.encoder_module.parameters()

    def decoder_params(self):
        return self.decoder_module.parameters()

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

    def get_x_graph(self, data, attr):
        x = getattr(data, attr)
        return x.view(data.num_graphs, -1)

    def encoder(self, X, edge_index, edge_attr=None, return_mean=False, **kwargs):
        logits = self.encoder_module(X, edge_index, edge_attr=edge_attr, **kwargs)
        if return_mean:
            mean, qz_x = self.likelihood_z(logits, return_mean=True)
            return mean, qz_x
        else:
            qz_x = self.likelihood_z(logits)
            return qz_x

    def sample_encoder(self, X, edge_index, edge_attr=None):
        qz_x = self.encoder(X, edge_index, edge_attr=edge_attr)
        sampled_z = qz_x.rsample()
        return sampled_z

    def decoder(self, Z, edge_index, edge_attr=None, return_type=None, **kwargs):
        logits = self.decoder_module(Z, edge_index, edge_attr, **kwargs)
        if return_type == 'mean':
            mean, px_z = self.likelihood_x(logits, return_mean=True)
            return mean, px_z
        elif return_type == 'sample':
            mean, px_z = self.likelihood_x(logits, return_mean=True)
            return px_z.sample(), px_z
        else:
            px_z = self.likelihood_x(logits)
            return px_z

    def sample_decoder(self, Z, adj):
        px_z = self.decoder(Z, adj)
        x_hat = px_z.rsample()
        return x_hat

    def compute_log_w(self, data,
                      K,
                      mask=None):
        """
        IWAE:  log(1\K \sum_k w_k) w_k = p(x, z_i)/ q(z_i | x)
            log_wi = log  p(x, z_i) - log q(z_i | x)
        Args:
            data:
            K:
            mask:

        Returns:

        """

        x_input = data.x.clone()

        if mask is not None:
            x_input[~mask] = 0.0

        log_w = []
        for k in range(K):
            qz_x = self.encoder(x_input, data.edge_index, edge_attr=data.edge_attr, node_ids=data.node_ids)
            z = qz_x.rsample()

            px_z_k = self.decoder(z, data.edge_index, edge_attr=data.edge_attr, node_ids=data.node_ids)
            log_prob_qz_x = qz_x.log_prob(z).sum(-1)  # Summing over dim(z)
            log_prob_pz = self.z_prior_distr.log_prob(z).sum(-1)
            log_prob_px_z = px_z_k.log_prob(data.x).sum(-1)

            log_w_k = log_prob_px_z + log_prob_pz - log_prob_qz_x

            if mask is not None:
                log_w.append(log_w_k[mask])
            else:
                log_w.append(log_w_k)

        log_w = torch.stack(log_w, dim=0)

        # [K, N]
        return log_w.T

    def compute_log_w_dreg(self, data, K):
        """
        IWAE dreg:  log(1\K \sum_k w_k) w_k = p(x, z_i)/ q(z_i | x)
            log_wi = log  p(x, z_i) - log q(z_i | x)
        Args:
            data:
            K:

        Returns:

        """

        log_w = []
        zs = []
        for k in range(K):
            qz_x = self.encoder(data.x, data.edge_index, edge_attr=data.edge_attr, node_ids=data.node_ids)
            z = qz_x.rsample()

            px_z_k = self.decoder(z, data.edge_index, edge_attr=data.edge_attr, node_ids=data.node_ids)

            qz_x_ = qz_x.__class__(qz_x.loc.detach(), qz_x.scale.detach())  # only difference to compute_log_w
            log_prob_qz_x = qz_x_.log_prob(z).sum(-1)
            log_prob_pz = self.z_prior_distr.log_prob(z).sum(-1)
            log_prob_px_z = px_z_k.log_prob(data.x).sum(-1)

            log_w_k = log_prob_px_z + log_prob_pz - log_prob_qz_x
            log_w.append(log_w_k)
            zs.append(z)

        log_w = torch.stack(log_w, dim=0)
        zs = torch.stack(zs, dim=0)
        # [K, N]
        return log_w.T, zs

    @torch.no_grad()
    def sample(self, adj, Z=None, n_samples=None):
        if (Z is None) == (n_samples is None):
            raise ValueError("Either `Z` or `n_samples` must be specified, but not both.")

        if Z is None:
            if not isinstance(n_samples, list): n_samples = [n_samples]
            Z = self.z_prior_distr.sample(n_samples)

        px_z = self.decoder(Z, adj)
        x_hat = px_z.rsample()
        return x_hat

    def forward(self, data, estimator, beta=1.0):
        x_input = data.x.clone()
        mask = None

        if estimator == 'elbo':

            qz_x = self.encoder(x_input,
                                data.edge_index,
                                edge_attr=data.edge_attr,
                                node_ids=data.node_ids)
            z = qz_x.rsample()

            px_z = self.decoder(z, data.edge_index, edge_attr=data.edge_attr, node_ids=data.node_ids)

            log_prob_x = px_z.log_prob(data.x).flatten(1).sum(1).mean()
            kl_z = torch.distributions.kl.kl_divergence(qz_x, self.z_prior_distr).flatten(1).sum(1).mean()

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
    def reconstruct(self, data,
                    use_mean_encoder=True,
                    return_type='mean'):
        z_mean, qz_x = self.encoder(data.x, data.edge_index, edge_attr=data.edge_attr,
                                    return_mean=True, node_ids=data.node_ids)

        z = z_mean if use_mean_encoder else qz_x.rsample()
        x_hat, _ = self.decoder(z, data.edge_index, edge_attr=data.edge_attr,
                                return_type=return_type, node_ids=data.node_ids)

        return z_mean, x_hat
