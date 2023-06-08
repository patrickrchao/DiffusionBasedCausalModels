import torch
import torch.nn as nn

from utils.constants import Cte
from utils.likelihoods import get_likelihood

from modules.dense import MLPModule


class CVAEModule(nn.Module):
    '''
    Conditional VAE
    '''

    def __init__(self, x_dim,
                 h_dim_list_dec,
                 h_dim_list_enc,
                 z_dim,
                 cond_dim=0,
                 act_name=Cte.RELU,
                 drop_rate=0.0,
                 likelihood_x=None,
                 distr_z='normal'):
        super(CVAEModule, self).__init__()

        likelihood_z = get_likelihood(distr_z, z_dim)

        c_list = [x_dim + cond_dim]
        c_list.extend(h_dim_list_enc)
        c_list.append(likelihood_z.params_size)

        self.encoder_module = MLPModule(h_dim_list=c_list,
                                        activ_name=act_name,
                                        bn=False,
                                        drop_rate=drop_rate)

        c_list = [z_dim + cond_dim]
        c_list.extend(h_dim_list_dec)
        c_list.append(likelihood_x.params_size)
        # Instantiate Decoder module

        self.decoder_module = MLPModule(h_dim_list=c_list,
                                        activ_name=act_name,
                                        bn=False,
                                        drop_rate=drop_rate)

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

    def encoder(self, X, cond_data=None, return_mean=False, **kwargs):
        if cond_data is not None:
            X = torch.cat([X, cond_data], dim=1)
        logits = self.encoder_module(X)
        if return_mean:
            mean, qz_x = self.likelihood_z(logits, return_mean=True)
            return mean, qz_x
        else:
            qz_x = self.likelihood_z(logits)
            return qz_x

    def decoder(self, Z, cond_data=None, return_type=None, **kwargs):
        if cond_data is not None:
            Z = torch.cat([Z, cond_data], dim=1)
        logits = self.decoder_module(Z, **kwargs)
        if return_type == 'mean':
            mean, px_z = self.likelihood_x(logits, return_mean=True)
            return mean, px_z
        elif return_type == 'sample':
            mean, px_z = self.likelihood_x(logits, return_mean=True)
            return px_z.sample(), px_z
        else:
            px_z = self.likelihood_x(logits)
            return px_z

    def compute_log_w(self, X, K, cond_data=None, mask=None):
        """
        IWAE log(1\K \sum_k w_k) w_k = p(x, z_i)/ q(z_i | x)
            log_wi = log  p(x, z_i) - log q(z_i | x)
        Args:
            X:
            K:
            cond_data:
            mask:

        Returns:

        """

        x_input = X.clone()

        if mask is not None:
            x_input[~mask] = 0.0

        log_w = []
        for k in range(K):
            qz_x = self.encoder(x_input, cond_data=cond_data)
            z = qz_x.rsample()
            px_z_k = self.decoder(z, cond_data=cond_data)

            log_prob_qz_x = qz_x.log_prob(z).sum(-1)  # Summing over dim(z)
            log_prob_pz = self.z_prior_distr.log_prob(z).sum(-1)
            log_prob_px_z = px_z_k.log_prob(X).sum(-1)

            log_w_k = log_prob_px_z + log_prob_pz - log_prob_qz_x

            if mask is not None:
                log_w.append(log_w_k[mask])
            else:
                log_w.append(log_w_k)

        log_w = torch.stack(log_w, dim=0)

        # [K, N]
        return log_w.T

    def forward(self, X, estimator, cond_data=None, beta=1.0):

        if estimator == 'elbo':

            qz_x = self.encoder(X, cond_data=cond_data)
            z = qz_x.rsample()

            px_z = self.decoder(z, cond_data=cond_data)
            log_prob_x = px_z.log_prob(X).flatten(1).sum(1).mean()
            kl_z = torch.distributions.kl.kl_divergence(qz_x, self.z_prior_distr).flatten(1).sum(1).mean()

            elbo = log_prob_x - beta * kl_z

            data = {'log_prob_x': log_prob_x,
                    'kl_z': kl_z}

            return elbo, data
        else:
            raise NotImplementedError

    @torch.no_grad()
    def reconstruct(self, X, cond_data=None):
        z_mean, qz_x = self.encoder(X, cond_data=cond_data, return_mean=True)

        x_hat, _ = self.decoder(z_mean, cond_data=cond_data, return_type='mean')

        return z_mean, x_hat
