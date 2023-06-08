"""
Code build from https://github.com/amirhk/recourse
"""
import json
import os
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import FunctionTransformer

import models._density_estimators as estims
from models.multicvae.cvae_module import CVAEModule
from utils.args_parser import mkdir
from utils.constants import Cte
from utils.metrics.time import Timer
from utils.optimizers import get_optimizer, get_scheduler


class MCVAE(pl.LightningModule):
    '''
    Multiple Conditional Variational Autoencoders
    '''

    def __init__(self,
                 h_dim_list_dec,
                 h_dim_list_enc,
                 z_dim,
                 drop_rate=0.0,
                 act_name=Cte.RELU,
                 likelihood_x=None,
                 distr_z='normal',
                 num_epochs_per_nodes=50,
                 topological_node_dims=[],  # Lists of dimensions are ordered by topology
                 topological_parents=[],
                 scaler=None
                 ):

        super(MCVAE, self).__init__()
        self.save_hyperparameters()

        self.num_nodes = len(topological_node_dims)

        self.random_train_sampler = None

        # added
        if scaler is None:
            self.scaler = FunctionTransformer(func=lambda x: x,
                                              inverse_func=lambda x: x)
        else:
            self.scaler = scaler

        self.num_epochs_per_nodes = num_epochs_per_nodes

        self.topological_node_dims = topological_node_dims
        self.topological_parents = topological_parents
        self.cvae_list = torch.nn.ModuleList()

        self.likelihoods_list = []
        for idx, node_dims in enumerate(self.topological_node_dims):
            if isinstance(likelihood_x, list):
                assert len(likelihood_x[idx]) == 1, 'Node with more than one likelihood not ipmlemented'
                lik_x = likelihood_x[idx][0]
            else:
                lik_x = likelihood_x

            self.likelihoods_list.append(lik_x)
            cvae = CVAEModule(
                x_dim=lik_x.domain_size,
                h_dim_list_dec=h_dim_list_dec,
                h_dim_list_enc=h_dim_list_enc,
                z_dim=z_dim,
                drop_rate=drop_rate,
                cond_dim=len(self.topological_parents[idx]),
                act_name=act_name,
                likelihood_x=lik_x,
                distr_z=distr_z
            )
            cvae.set_z_prior_distr(self.device)
            self.cvae_list.append(cvae)

        self.z_dim = z_dim
        self.timer = Timer()

        self.save_dir = None

        self.my_evaluator = None

    def set_my_evaluator(self, evaluator):
        self.my_evaluator = evaluator

    def get_x_graph(self, data, attr):
        x = getattr(data, attr)
        return x.view(data.num_graphs, -1)

    def monitor(self):
        return 'valid_iwae_100'

    def monitor_mode(self):
        return 'max'

    def set_random_train_sampler(self, sampler):
        self.random_train_sampler = sampler

    def get_x_graph(self, data, attr):
        x = getattr(data, attr)
        if attr in ['x', 'x_i']:
            x = x[data.mask]
        return x.view(data.num_graphs, -1)

    @torch.no_grad()
    def samples_aggregated_posterior(self, num_samples, idx):
        batch = self.random_train_sampler(num_samples)
        X = self.get_x_graph(batch, 'x')

        x = X[:, self.topological_node_dims[idx]]  # shape [1000, 1]
        if len(self.topological_parents[idx]) == 0:  # Root node
            pa = None
        else:
            pa = X[:, self.topological_parents[idx]]  # shape [1000, 1]

        q_z_x = self.cvae_list[idx].encoder(x, cond_data=pa, return_mean=False)

        return q_z_x.sample()

    def set_optim_params(self, optim_params, sched_params):
        self.optim_params = optim_params
        self.sched_params = sched_params

    def configure_optimizers(self):
        scheduler_list = []
        optimizer_list = []
        for idx, node_dims in enumerate(self.topological_node_dims):

            optimizer = get_optimizer(self.optim_params['name'])(self.cvae_list[idx].parameters(),
                                                                 **self.optim_params['params'])
            optimizer_list.append(optimizer)

            if isinstance(self.sched_params, dict):
                scheduler = get_scheduler(self.sched_params['name'])(optimizer, **self.sched_params['params'])
                scheduler_list.append(scheduler)
            idx += 1
        return optimizer_list, scheduler_list

    def is_training_node_i(self, node_id, current_epoch):
        if ((current_epoch >= self.num_epochs_per_nodes * (node_id)) and (
                current_epoch < self.num_epochs_per_nodes * (node_id + 1))):
            return True
        else:
            return False

    def forward(self, data):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        X = self.get_x_graph(batch, 'x')  # shape [1000, 3] # works because 1 Dim X
        # note, only learning cvae for non-root nodes

        for i in range(self.num_nodes):
            if optimizer_idx == i and self.is_training_node_i(i, self.current_epoch):  # train node 0
                x = X[:, self.topological_node_dims[i]]  # shape [1000, 1]
                if len(self.topological_parents[i]) == 0:  # Root node
                    pa = None
                else:
                    pa = X[:, self.topological_parents[i]]  # shape [1000, 1]

                objective, data = self.cvae_list[i](x, estimator='elbo', cond_data=pa)
                self.log(f'train_objective_{optimizer_idx}', objective.item(), prog_bar=True)
                for key, value in data.items():
                    self.log(f'train_{key}_{i}', value.item(), prog_bar=True)

                return -objective

    def on_train_epoch_start(self) -> None:
        self.timer.tic('train')

    def on_train_epoch_end(self, outputs) -> None:
        time = self.timer.toc('train')
        self.logger.experiment.add_scalar('train_time', time, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        output = {}

        K = 100
        metrics = self.get_objective_metrics_batch(batch, K=K)

        for key, value in metrics.items():
            output[f'valid_{key}'] = value[0]
            self.log(f'valid_{key}', value[0], prog_bar=True)

        return output[f'valid_iwae_{K}']

    def on_validation_epoch_start(self) -> None:
        self.timer.stop('train_total')

    def on_validation_epoch_end(self) -> None:
        self.timer.resume('train_total')

    def on_test_epoch_start(self) -> None:
        self.x_test = []
        self.x_hat = []
        return

    def test_step(self, batch, batch_idx):
        output = {}
        metrics = self.get_objective_metrics_batch(batch, K=20)

        for key, value in metrics.items():
            output[f'test_{key}'] = value

        self.log(output, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        return

    def on_epoch_end(self) -> None:
        self.eval()
        # log sampled images

        if self.current_epoch % 100 == 0:
            self.my_evaluator.set_save_dir(self.logger.save_dir)
            self.my_evaluator.set_logger(self.logger)
            self.my_evaluator.set_current_epoch(self.current_epoch)
            self.my_evaluator.complete_logs(data_loader=self.test_dataloader(), name='test', plots=False)

        self.train()  # IMPORTANT: Set back to train mode!
        return

    def on_fit_end(self):
        with open(os.path.join(self.logger.save_dir, 'time.json'), 'w') as f:
            toc = self.timer.toc('train_total')
            ouput = {'train_time_total': toc,
                     'train_epochs': self.current_epoch,
                     'train_time_avg_per_epoch': toc / self.current_epoch}
            json.dump(ouput, f)

        self.my_evaluator.set_save_dir(self.logger.save_dir)
        self.my_evaluator.set_logger(self.logger)
        self.my_evaluator.set_current_epoch(self.current_epoch)
        self.my_evaluator.complete_logs(data_loader=self.test_dataloader(), name='test', plots=False)

        return

    def on_fit_start(self):
        self.eval()
        self.timer.tic('train_total')
        img_folder = mkdir(os.path.join(self.logger.save_dir, 'images'))
        for i in range(self.num_nodes):
            self.cvae_list[i].set_z_prior_distr(self.device)  # Just to move the prior to GPU if needed

    @torch.no_grad()
    def get_objective_metrics(self, data_loader, name, K=100):
        output = {}

        for idx, batch in enumerate(iter(data_loader)):
            metrics = self.get_objective_metrics_batch(batch, K=K)

        for key, value in metrics.items():
            output[f'{name}_{key}'] = np.mean(value)

        return output

    @torch.no_grad()
    def get_objective_metrics_batch(self, batch, K=100):
        metrics = {'elbo': [], f'iwae_{K}': []}
        X = self.get_x_graph(batch, 'x')
        for i in range(self.num_nodes):
            x = X[:, self.topological_node_dims[i]]  # shape [1000, 1]
            if len(self.topological_parents[i]) == 0:  # Root node
                pa = None
            else:
                pa = X[:, self.topological_parents[i]]  # shape [1000, 1]

            objective, data = self.cvae_list[i](x, cond_data=pa, estimator='elbo', beta=1)
            objective = objective.unsqueeze(0).unsqueeze(1)

            log_w = self.cvae_list[i].compute_log_w(x, cond_data=pa, K=K)
            iwae_10, _ = estims.IWAE(log_w, trick=False)
            iwae_objective = iwae_10.mean().unsqueeze(0).unsqueeze(1)

            if i == 0:
                Objective = objective
                IWAE10 = iwae_objective
            else:
                Objective = torch.cat((Objective, objective), dim=-1)
                IWAE10 = torch.cat((IWAE10, iwae_objective), dim=-1)

        # mean across all cvae
        av_objective = torch.mean(Objective)  # mean of each row
        metrics['elbo'].append(av_objective.squeeze().squeeze().item())

        # mean across all cvae
        av_iwae_10 = torch.mean(IWAE10)  # mean of each row
        metrics[f'iwae_{K}'].append(av_iwae_10.squeeze().squeeze().item())
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader, name='test', save_dir=None, plots=False):

        self.my_evaluator.set_save_dir(save_dir if save_dir is not None else self.logger.save_dir, )
        self.my_evaluator.set_logger(self.logger)
        self.my_evaluator.set_current_epoch(100000)

        output = self.my_evaluator.evaluate(dataloader, name=name, plots=plots)
        return output

    def get_intervention_node_idx(self, x_I, node_names_list):
        node_name = list(x_I.keys())[0]
        return node_names_list.index(node_name)

    @torch.no_grad()
    def get_interventional_distr(self, data_loader,
                                 x_I: Dict[str, float],
                                 use_aggregated_posterior=False,
                                 normalize=True):

        """
        Get x generated distribution  w/o intervention or with diagonal adjacency.
        Args:
            data_loader:
            x_I:
                If x_I is None compute the distribution of the original SCM, if x_I is a dict
                then compute the interventional distribution. E.g. x_I = {'x1': 0} computes the
                interventional distribution with do(x1=0)
            use_aggregated_posterior:
            normalize:

        Returns:
            x_gener_dict_out: dict of torch.Tensor
                Generated distribution
            x_real_dict_out: dict of torch.Tensor
                distribution of the dataset (real data)
        """
        assert isinstance(x_I, dict)
        data_loader.dataset.set_intervention(x_I)
        iterator = iter(data_loader)
        self.eval()
        x_gener_dict = {'intervened': [],
                        'parents': [],
                        'children': [],
                        'all': []}
        x_real_dict = {'intervened': [],
                       'parents': [],
                       'children': [],
                       'all': []}

        inter_idx = self.get_intervention_node_idx(x_I, data_loader.dataset.nodes_list)

        for idx, batch in enumerate(iterator):
            X_i = self.get_x_graph(batch, 'x')
            X_int_pa = None
            for i in range(self.num_nodes):
                x_i = X_i[:, self.topological_node_dims[i]]  # shape [1000, 1]

                if len(self.topological_parents[i]) == 0:  # Root node
                    pa = None
                else:
                    pa = X_int_pa[:, self.topological_parents[i]]  # shape [1000, 1]

                if i == inter_idx:  # Intervened node
                    if i == 0:
                        X_int_pa = x_i
                    else:
                        X_int_pa = torch.cat((X_int_pa, x_i), dim=-1)

                else:
                    if use_aggregated_posterior:
                        z = self.samples_aggregated_posterior(num_samples=batch.num_graphs, idx=i).to(self.device)
                    else:
                        z = self.cvae_list[i].z_prior_distr.sample([batch.num_graphs]).to(self.device)  # dim [1000, 4]

                    x_rec, _ = self.cvae_list[i].decoder(z, cond_data=pa, return_type='sample')

                    if i == 0:
                        X_int_pa = x_rec
                    else:
                        X_int_pa = torch.cat((X_int_pa, x_rec), dim=-1)

            # Not normalized
            x_inter, set_nodes = data_loader.dataset.sample_intervention(x_I=x_I,
                                                                         n_samples=batch.num_graphs,
                                                                         return_set_nodes=True)
            # normalized
            # x_inter = torch.tensor(self.scaler.transform(x_inter))
            x_inter = torch.tensor(x_inter)
            X_int_pa = self.scaler.inverse_transform(X_int_pa)
            x_gener_dict['parents'].append(X_int_pa[:, set_nodes['parents']])
            x_gener_dict['intervened'].append(X_int_pa[:, set_nodes['intervened']])
            x_gener_dict['children'].append(X_int_pa[:, set_nodes['children']])
            x_gener_dict['all'].append(X_int_pa)

            x_real_dict['parents'].append(x_inter[:, set_nodes['parents']])
            x_real_dict['intervened'].append(x_inter[:, set_nodes['intervened']])
            x_real_dict['children'].append(x_inter[:, set_nodes['children']])
            x_real_dict['all'].append(x_inter)

        x_gener_dict_out = {}
        x_real_dict_out = {}
        for key, values in x_gener_dict.items():
            x_gener_dict_out[key] = torch.cat(values)
            x_real_dict_out[key] = torch.cat(x_real_dict[key])

        data_loader.dataset.clean_intervention()

        return x_gener_dict_out, x_real_dict_out

    @torch.no_grad()
    def get_observational_distr(self, data_loader,
                                use_links=True,
                                use_aggregated_posterior=False,
                                normalize=True):
        """
        Get x generated (observational) distribution  w/o intervention or with diagonal adjacency.
        Args:
            data_loader:
            use_links:
                If false, then uses an diagonal adjacency matrix to compute the distribution
            use_aggregated_posterior:
            normalize:

        Returns:
            z_list: torch.Tensor
                Latent code of the generated distribution
            x: torch.Tensor
                Generated distribution
            x_real: torch.Tensor
                distribution of the dataset (real data)
        """
        # if use_links is False:
        #     data_loader.dataset.diagonal_SCM()
        iterator = iter(data_loader)
        self.eval()
        x, z_list = [], []
        x_real = []

        for idx, batch in enumerate(iterator):

            # sample

            for i in range(self.num_nodes):
                if use_aggregated_posterior:
                    z = self.samples_aggregated_posterior(num_samples=batch.num_graphs, idx=i).to(self.device)
                else:
                    z = self.cvae_list[i].z_prior_distr.sample([batch.num_graphs]).to(self.device)  # dim [1000, 4]

                if len(self.topological_parents[i]) == 0:  # Root node
                    pa = None
                else:
                    pa = X_hat_all[:, self.topological_parents[i]]  # shape [1000, 1]

                x_hat, _ = self.cvae_list[i].decoder(z, cond_data=pa, return_type='sample')

                if i == 0:
                    X_hat_all = x_hat
                    Z_all = z
                else:
                    X_hat_all = torch.cat((X_hat_all, x_hat), dim=-1)
                    Z_all = torch.cat((Z_all, z), dim=-1)

            if normalize:
                x_real.append(self.get_x_graph(batch, 'x'))
                x.append(X_hat_all.view(batch.num_graphs, -1))
            else:
                x_real.append(self.scaler.inverse_transform(self.get_x_graph(batch, 'x')))
                x.append(self.scaler.inverse_transform(X_hat_all.view(batch.num_graphs, -1)))

            z_list.append(Z_all.view(batch.num_graphs, -1))

        data_loader.dataset.clean_intervention()

        return torch.cat(z_list), torch.cat(x), torch.cat(x_real)

    def compute_counterfactual(self, batch, x_I):
        raise NotImplementedError

    @torch.no_grad()
    def compute_counterfactual(self, batch, x_I):

        X_i = batch.x_i.view(batch.num_graphs, -1)
        for i in range(self.num_nodes):
            x_i = X_i[:, self.topological_node_dims[i]]  # shape [1000, 1]

            if len(self.topological_parents[i]) == 0:  # Root node
                pa = None
            else:
                pa = X_cf_pa[:, self.topological_parents[i]]  # shape [1000, 1]

            z_cf, _ = self.cvae_list[i].encoder(x_i, cond_data=pa, return_mean=True)
            x_CF, _ = self.cvae_list[i].decoder(z_cf, cond_data=pa, return_type='sample')

            int_nodes = self.get_int_nodes(x_I)
            if i in int_nodes:
                if i == 0:
                    X_cf_pa = x_i
                    X_cf_all = x_CF
                    Z_cf = z_cf
                else:
                    X_cf_pa = torch.cat((X_cf_pa, x_i), dim=-1)
                    X_cf_all = torch.cat((X_cf_all, x_CF), dim=-1)
                    Z_cf = torch.cat((Z_cf, z_cf), dim=-1)
            else:
                if i == 0:
                    X_cf_pa = x_CF
                    X_cf_all = x_CF
                    Z_cf = z_cf
                else:
                    X_cf_pa = torch.cat((X_cf_pa, x_CF), dim=-1)
                    X_cf_all = torch.cat((X_cf_all, x_CF), dim=-1)
                    Z_cf = torch.cat((Z_cf, z_cf), dim=-1)

        return X_cf_all.view(batch.num_graphs, -1), None, Z_cf.reshape(
            batch.num_graphs, -1)

    @torch.no_grad()
    def get_counterfactual_distr(self, data_loader,
                                 x_I=None,
                                 is_noise=False,
                                 normalize=True):

        assert isinstance(x_I, dict)
        data_loader.dataset.set_intervention(x_I, is_noise=is_noise)
        iterator = iter(data_loader)
        self.eval()

        x_gener_dict = {'intervened': [],
                        'children': [],
                        'all': []}
        x_real_dict = {'intervened': [],
                       'children': [],
                       'all': []}

        x_factual_dict = {'all': []}
        inter_idx = self.get_intervention_node_idx(x_I, data_loader.dataset.nodes_list)

        for idx, batch in enumerate(iterator):
            X_i = batch.x_i.view(batch.num_graphs, -1)
            X = batch.x.view(batch.num_graphs, -1)
            X_cf_pa = None
            for i in range(self.num_nodes):
                x_i = X_i[:, self.topological_node_dims[i]]  # shape [1000, 1]

                if len(self.topological_parents[i]) == 0:  # Root node
                    pa_cf = None
                    pa_f = None
                else:
                    pa_cf = X_cf_pa[:, self.topological_parents[i]]  # shape [1000, 1]
                    pa_f = X[:, self.topological_parents[i]]

                if i == inter_idx:
                    if i == 0:
                        X_cf_pa = x_i
                    else:
                        X_cf_pa = torch.cat((X_cf_pa, x_i), dim=-1)
                else:
                    z_cf, _ = self.cvae_list[i].encoder(x_i, cond_data=pa_f, return_mean=True)
                    x_CF, _ = self.cvae_list[i].decoder(z_cf, cond_data=pa_cf, return_type='sample')

                    if i == 0:
                        X_cf_pa = x_CF
                    else:
                        X_cf_pa = torch.cat((X_cf_pa, x_CF), dim=-1)

            # Not normalized
            x_cf_real, set_nodes = data_loader.dataset.get_counterfactual(
                x_factual=self.scaler.inverse_transform(self.get_x_graph(batch, 'x')),
                u_factual=batch.u.view(batch.num_graphs, -1),
                x_I=x_I,
                is_noise=is_noise,
                return_set_nodes=True)

            # x_cf_real = self.scaler.transform(x_cf_real)
            if normalize:
                x_cf_real = self.scaler.transform(x_cf_real)
            else:
                x_cf_real = torch.tensor(x_cf_real)
                X_cf_pa = self.scaler.inverse_transform(X_cf_pa)

            x_gener_dict['intervened'].append(X_cf_pa[:, set_nodes['intervened']])
            x_gener_dict['children'].append(X_cf_pa[:, set_nodes['children']])
            x_gener_dict['all'].append(X_cf_pa)

            x_real_dict['intervened'].append(x_cf_real[:, set_nodes['intervened']])
            x_real_dict['children'].append(x_cf_real[:, set_nodes['children']])
            x_real_dict['all'].append(x_cf_real)

            if normalize:
                x_factual_dict['all'].append(self.get_x_graph(batch, 'x'))
            else:
                x_factual_dict['all'].append(self.scaler.inverse_transform(self.get_x_graph(batch, 'x')))

        x_gener_dict_out = {}
        x_real_dict_out = {}
        x_factual_dict_out = {}

        for key, values in x_gener_dict.items():
            x_gener_dict_out[key] = torch.cat(values)
            x_real_dict_out[key] = torch.cat(x_real_dict[key])
        for key, values in x_factual_dict.items():
            x_factual_dict_out[key] = torch.cat(values)

        data_loader.dataset.clean_intervention()

        return x_gener_dict_out, x_real_dict_out, x_factual_dict_out

    @torch.no_grad()
    def get_x(self, data_loader):
        iterator = iter(data_loader)
        self.eval()
        x = []
        for idx, batch in enumerate(iterator):
            x.append(batch.x.view(batch.num_graphs, -1))
        return torch.cat(x)

    @torch.no_grad()
    def get_reconstruction_distr(self, data_loader,
                                 normalize=True):

        """
        Reconstruct all the feates of all the  graphs in data loader, i.e.,
        Z \sim q(Z|X, A) and X_hat p(X | Z, A)
        Args:
            data_loader:
            normalize:

        Returns:
            z_list: torch.Tensor
                Latent code of the reconstructed distribution, i.e. q(z|x)
            x: torch.Tensor
                reconstructed samples
            x_real: torch.Tensor
                original  samples (real data)
        """

        iterator = iter(data_loader)
        self.eval()
        x_rec_list, z_list = [], []
        x_real = []
        for idx, batch in enumerate(iterator):
            X = self.get_x_graph(batch, 'x')
            for i in range(self.num_nodes):
                x = X[:, self.topological_node_dims[i]]  # shape [1000, 1]
                if len(self.topological_parents[i]) == 0:  # Root node
                    pa = None
                else:
                    pa = X[:, self.topological_parents[i]]  # shape [1000, 1]

                z_hat, x_rec = self.cvae_list[i].reconstruct(x, cond_data=pa)

                if i == 0:
                    X_rec_all = x_rec
                    Z_all = z_hat
                else:
                    X_rec_all = torch.cat((X_rec_all, x_rec), dim=-1)
                    Z_all = torch.cat((Z_all, z_hat), dim=-1)

            if normalize:
                x_rec_list.append(X_rec_all.view(batch.num_graphs, -1))
                x_real.append(X)
            else:
                x_rec_list.append(self.scaler.inverse_transform(X_rec_all.view(batch.num_graphs, -1)))
                x_real.append(self.scaler.inverse_transform(X))

            z_list.append(Z_all.reshape(batch.num_graphs, -1))

        return torch.cat(z_list), torch.cat(x_rec_list), torch.cat(x_real)

    @torch.no_grad()
    def get_obs_distribution(self, data_loader):
        iterator = iter(data_loader)
        self.eval()
        x = []
        for idx, batch in enumerate(iterator):
            x.append(batch.x.view(batch.num_graphs, -1))

        return torch.cat(x)
