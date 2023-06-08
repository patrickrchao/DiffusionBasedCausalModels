import json
import os
from typing import List, Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import FunctionTransformer

import models._density_estimators as estims
from utils.args_parser import mkdir
from utils.constants import Cte
from utils.dropout import dropout_adj, dropout_adj_parents
from utils.metrics.time import Timer
from utils.optimizers import get_optimizer, get_scheduler


class VACA(pl.LightningModule):
    """
    VACA Lightning Module
    """

    def __init__(self,
                 h_dim_list_dec: List[int],
                 h_dim_list_enc: List[int],
                 z_dim: int,
                 m_layers: int = 1,
                 deg: List[float] = None,  # Only PNA architecture
                 edge_dim: int = None,
                 num_nodes: int = None,
                 beta: float = 1.0,
                 annealing_beta: bool = False,
                 residual: int = 0,  # Only PNA architecture
                 drop_rate: float = 0.0,
                 dropout_adj_rate: float = 0.0,
                 dropout_adj_pa_rate: float = 0.0,
                 dropout_adj_pa_prob_keep_self: float = 0.0,
                 keep_self_loops: bool = True,
                 dropout_adj_T: int = 0,  # Epoch to start the dropout_adj_T
                 act_name: str = Cte.RELU,
                 likelihood_x: Any = None,  # Heterogneous: List[List[BaseLikelihood]] || Simple: BaseLikelihood
                 distr_z: str = 'normal',
                 architecture: str = None,  # PNA, DGNN, DPNA
                 estimator: str = 'iwaedreg',
                 K=1,  # Only for IWAE estimator
                 scaler: Any = None,
                 init: str = None,
                 is_heterogeneous: bool = False,
                 norm_categorical: bool = False,
                 norm_by_dim: bool = False,
                 ):
        super(VACA, self).__init__()
        assert init is None, 'Only default init is implemented'

        self.save_hyperparameters()
        self.estimator = estimator

        self.num_nodes = num_nodes
        self.keep_self_loops = keep_self_loops

        self.random_train_sampler = None

        if scaler is None:
            self.scaler = FunctionTransformer(func=lambda x: x,
                                              inverse_func=lambda x: x)
        else:
            self.scaler = scaler

        self.beta = beta
        self.annealing_beta = annealing_beta

        if is_heterogeneous:
            from models.vaca.hvaca_module import HVACAModule

            self.model = HVACAModule(likelihoods_x=likelihood_x,
                                     h_dim_list_dec=h_dim_list_dec,  # Hidden layers in the generative network
                                     h_dim_list_enc=h_dim_list_enc,  # Hidden layers in the inference network
                                     z_dim=z_dim,
                                     m_layers=m_layers,
                                     deg=deg,
                                     edge_dim=edge_dim,
                                     residual=residual,
                                     drop_rate=drop_rate,
                                     act_name=act_name,
                                     distr_z=distr_z,
                                     architecture=architecture,
                                     norm_categorical=norm_categorical,
                                     norm_by_dim=norm_by_dim,
                                     K=K
                                     )
        else:
            from models.vaca.vaca_module import VACAModule
            x_dim = likelihood_x.domain_size

            self.model = VACAModule(x_dim=x_dim,
                                    h_dim_list_dec=h_dim_list_dec,  # Hidden layers in the generative network
                                    h_dim_list_enc=h_dim_list_enc,  # Hidden layers in the inference network
                                    z_dim=z_dim,
                                    m_layers=m_layers,
                                    deg=deg,
                                    edge_dim=edge_dim,
                                    num_nodes=num_nodes,
                                    residual=residual,
                                    drop_rate=drop_rate,
                                    act_name=act_name,
                                    likelihood_x=likelihood_x,
                                    distr_z=distr_z,
                                    architecture=architecture,
                                    K=K
                                    )

        self.is_heterogeneous = is_heterogeneous

        self.model.set_z_prior_distr(self.device)
        self.z_dim = z_dim
        self.timer = Timer()

        self.dropout_adj = dropout_adj_rate
        self.dropout_adj_pa_prob_keep_self = dropout_adj_pa_prob_keep_self
        self.dropout_adj_pa = dropout_adj_pa_rate
        self.dropout_adj_T = dropout_adj_T

        self.save_dir = None

        self.my_evaluator = None

    def set_my_evaluator(self, evaluator):
        self.my_evaluator = evaluator

    def monitor(self):
        return 'valid_iwae_100'

    def monitor_mode(self):
        return 'max'

    def set_random_train_sampler(self, sampler):
        self.random_train_sampler = sampler

    @torch.no_grad()
    def samples_aggregated_posterior(self, num_samples):
        batch = self.random_train_sampler(num_samples)
        # ADDED BY PATRICK CHAO 8/10/2022
        batch = batch.to(self.device)
        q_z_x = self.model.encoder(batch.x, batch.edge_index, edge_attr=batch.edge_attr,
                                   return_mean=False, node_ids=batch.node_ids)
        return q_z_x.sample()

    def get_x_graph(self, data, attr):
        return self.model.get_x_graph(data, attr)

    def set_optim_params(self, optim_params, sched_params):
        self.optim_params = optim_params
        self.sched_params = sched_params

    def configure_optimizers(self):
        optim = get_optimizer(self.optim_params['name'])(self.parameters(), **self.optim_params['params'])
        if isinstance(self.sched_params, dict):
            sched = get_scheduler(self.sched_params['name'])(optim, **self.sched_params['params'])
        else:
            sched = []
        return [optim], sched

    def forward(self, data, *args, **kwargs):
        return self.model(data, estimator=self.estimator)

    def get_beta_annealing_factor(self, current_epoch):
        if self.annealing_beta > 0:  # Do annealing
            return max(min((current_epoch - 10) / self.annealing_beta, 1.0), 0)
        else:
            return 1.0

    def training_step(self, batch, batch_idx):
        batch = batch.to(self.device)

        if self.dropout_adj > 0.0 and self.current_epoch >= self.dropout_adj_T:
            batch = batch.clone()
            batch.edge_index, batch.edge_attr = dropout_adj(batch.edge_index, batch.edge_attr,
                                                            p=self.dropout_adj, keep_self_loops=self.keep_self_loops)

        if self.dropout_adj_pa > 0.0 and self.current_epoch >= self.dropout_adj_T:
            batch = batch.clone()
            batch.edge_index, batch.edge_attr = dropout_adj_parents(batch.edge_index, batch.edge_attr,
                                                                    p=self.dropout_adj_pa,
                                                                    prob_keep_self=self.dropout_adj_pa_prob_keep_self)

        objective, data = self.model(batch,
                                     estimator=self.estimator,
                                     beta=self.beta * self.get_beta_annealing_factor(self.current_epoch))
        self.log('train_objective', objective.item(), prog_bar=True)
        for key, value in data.items():
            self.log(f'train_{key}', value.item(), prog_bar=True)
        return -objective

    def on_train_epoch_start(self) -> None:
        self.timer.tic('train')

    def on_train_epoch_end(self, outputs) -> None:
        time = self.timer.toc('train')
        self.logger.experiment.add_scalar('train_time', time, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        batch = batch.to(self.device)
        objective, data = self.model(batch, estimator='elbo', beta=1.0)
        self.log('valid_elbo', objective.item(), prog_bar=True)
        for key, value in data.items():
            self.log(f'valid_{key}', value.item(), prog_bar=True)

        K = 100
        log_w = self.compute_log_w(batch, K=K)
        iwae, _ = estims.IWAE(log_w[:, :K], trick=False)
        iwae = iwae.mean()
        self.log(f'valid_iwae_{K}', iwae.item(), prog_bar=True)

        return {f'valid_iwae_{K}': iwae.item()}

    def on_validation_epoch_start(self) -> None:
        self.timer.stop('train_total')

    def on_validation_epoch_end(self) -> None:
        self.timer.resume('train_total')

    def on_test_epoch_start(self) -> None:
        self.x_test = []
        self.x_hat = []
        return

    def test_step(self, batch, batch_idx):
        batch = batch.to(self.device)
        objective, data = self.model(batch, estimator='elbo', beta=1.0)
        self.log('test_elbo', objective.item(), prog_bar=True)
        for key, value in data.items():
            self.log(f'test_{key}', value.item(), prog_bar=True)

        K = 20
        log_w = self.compute_log_w(batch, K=K)
        iwae, _ = estims.IWAE(log_w, trick=False)
        iwae = iwae.mean()
        self.log(f'test_iwae_{K}', iwae.item(), prog_bar=True)

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
        self.model.set_z_prior_distr(self.device)  # Just to move the prior to GPU if needed

    @torch.no_grad()
    def get_objective_metrics(self,
                              data_loader,
                              name):
        output = {}

        K = 100

        metrics = {'elbo': [], f'iwae_{K}': []}
        for idx, batch in enumerate(iter(data_loader)):
            objective, data = self.model(batch, estimator='elbo', beta=1)
            metrics['elbo'].append(objective.item())
            log_w = self.compute_log_w(batch, K=K)
            iwae_10, _ = estims.IWAE(log_w, trick=False)
            metrics[f'iwae_{K}'].append(iwae_10.mean().item())

        for key, value in metrics.items():
            output[f'{name}_{key}'] = np.mean(value)

        return output

    @torch.no_grad()
    def evaluate(self,
                 dataloader,
                 name='test',
                 save_dir=None,
                 plots=False):
        self.my_evaluator.set_save_dir(save_dir if save_dir is not None else self.logger.save_dir, )
        self.my_evaluator.set_logger(self.logger)
        self.my_evaluator.set_current_epoch(100000)

        output = self.my_evaluator.evaluate(dataloader, name=name, plots=plots)
        return output

    def my_cf_fairness(self,
                       data_module,
                       save_dir=None):
        self.my_evaluator.set_save_dir(save_dir if save_dir is not None else self.logger.save_dir, )
        self.my_evaluator.set_logger(self.logger)
        self.my_evaluator.set_current_epoch(100000)

        output = self.my_evaluator.evaluate_cf_fairness(data_module)
        print(output)
        return output

    def compute_log_w(self, data, K):
        return self.model.compute_log_w(data, K=K)

    def compute_log_w_dreg(self, data, K):
        return self.model.compute_log_w_dreg(data, K=K)

    @torch.no_grad()
    def get_observational_distr(self, data_loader,
                                use_links: bool = True,
                                use_aggregated_posterior: bool = False,
                                num_batches: bool = None,
                                normalize: bool = True):

        """
        Get x generated distribution  w/o intervention or with diagonal adjacency.
        Parameters
        Args:
            data_loader:
            use_links:
                If false, then uses an diagonal adjacency matrix to compute the distribution
            use_aggregated_posterior:
            num_batches:
            normalize:

        Returns:
            z_list: torch.Tensor
                Latent code of the generated distribution
            x: torch.Tensor
                Generated distribution
            x_real: torch.Tensor
                distribution of the dataset (real data)
        """

        if use_links is False:
            data_loader.dataset.diagonal_SCM()
        iterator = iter(data_loader)
        self.eval()
        x, z_list = [], []
        x_real = []

        for idx, batch in enumerate(iterator):
            batch = batch.to(self.device)
            if isinstance(num_batches, int) and idx > num_batches: break
            if use_aggregated_posterior:
                z = self.samples_aggregated_posterior(num_samples=batch.num_graphs).to(self.device)
            else:
                z = self.model.z_prior_distr.sample([batch.num_nodes]).to(self.device)

            x_hat, _ = self.model.decoder(z, batch.edge_index, edge_attr=batch.edge_attr,
                                          return_type='sample', node_ids=batch.node_ids)
            if normalize:
                x_real.append(self.get_x_graph(batch, 'x'))
                x.append(x_hat.view(batch.num_graphs, -1))
            else:
                x_real.append(self.scaler.inverse_transform(self.get_x_graph(batch, 'x')))
                x.append(self.scaler.inverse_transform(x_hat.view(batch.num_graphs, -1)))

            z_list.append(z.view(batch.num_graphs, -1))

        data_loader.dataset.clean_intervention()

        return torch.cat(z_list), torch.cat(x), torch.cat(x_real)

    @torch.no_grad()
    def get_intervention(self, batch,
                         x_I,
                         nodes_list,
                         return_type: str = 'sample',
                         use_aggregated_posterior: bool = False,
                         normalize: bool = True):
        """
        Get x generated distribution  w/o intervention or with diagonal adjacency.
        Parameters
        Args:
            data_loader:
            x_I:
                If x_I is None compute the distribution of the original SCM, if x_I is a dict
                then compute the interventional distribution. E.g. x_I = {'x1': 0} computes the
                interventional distribution with do(x1=0)
            use_aggregated_posterior:
            num_batches:
            normalize:

        Returns:
            x_gener_dict_out: dict of torch.Tensor
                Generated distribution
            x_real_dict_out: dict of torch.Tensor
                distribution of the dataset (real data)
        """

        self.eval()

        if use_aggregated_posterior:
            z = self.samples_aggregated_posterior(num_samples=batch.num_graphs).to(self.device)
        else:
            z = self.model.z_prior_distr.sample([batch.num_nodes]).to(self.device)

        z = z.view(batch.num_graphs, -1)

        z_mean, _ = self.model.encoder(batch.x_i, batch.edge_index_i, edge_attr=batch.edge_attr_i,
                                       return_mean=True, node_ids=batch.node_ids)
        z_mean = z_mean.reshape(batch.num_graphs, -1)
        for node_name, _ in x_I.items():
            i = nodes_list.index(node_name)
            z[:, self.z_dim * i:self.z_dim * (i + 1)] = z_mean[:, self.z_dim * i:self.z_dim * (i + 1)]

        z = z.view(-1, self.z_dim)

        x_hat, _ = self.model.decoder(z, batch.edge_index_i, edge_attr=batch.edge_attr_i,
                                      return_type=return_type, node_ids=batch.node_ids)

        x_hat = x_hat.reshape(batch.num_graphs, -1)
        if not normalize:
            x_hat = self.scaler.inverse_transform(x_hat)

        return x_hat, z

    @torch.no_grad()
    def get_interventional_distr(self, data_loader,
                                 x_I: Dict[str, float],
                                 use_aggregated_posterior: bool = False,
                                 num_batches: int = None,
                                 normalize: bool = True):
        """
        Get x generated distribution  w/o intervention or with diagonal adjacency.
        Parameters
        Args:
            data_loader:
            x_I:
                If x_I is None compute the distribution of the original SCM, if x_I is a dict
                then compute the interventional distribution. E.g. x_I = {'x1': 0} computes the
                interventional distribution with do(x1=0)
            use_aggregated_posterior:
            num_batches:
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
                        'children': [],
                        'all': []}
        x_real_dict = {'intervened': [],
                       'children': [],
                       'all': []}

        for idx, batch in enumerate(iterator):
            batch = batch.to(self.device)
            if isinstance(num_batches, int) and idx > num_batches: break
            x_hat, z = self.get_intervention(batch=batch,
                                             x_I=data_loader.dataset.x_I,
                                             nodes_list=data_loader.dataset.nodes_list,
                                             return_type='sample',
                                             use_aggregated_posterior=False,
                                             normalize=normalize)
            x_inter, set_nodes = data_loader.dataset.sample_intervention(x_I=x_I,
                                                                         n_samples=batch.num_graphs,
                                                                         return_set_nodes=True)

            if x_inter is not None:
                if normalize:
                    x_inter = torch.tensor(self.scaler.transform(x_inter))
                else:
                    x_inter = torch.Tensor(x_inter)
                # x_real_dict['parents'].append(x_inter[:, set_nodes['parents']])
                x_real_dict['intervened'].append(x_inter[:, set_nodes['intervened']])
                x_real_dict['children'].append(x_inter[:, set_nodes['children']])
                x_real_dict['all'].append(x_inter)

            # x_gener_dict['parents'].append(x_hat[:, set_nodes['parents']])
            x_gener_dict['intervened'].append(x_hat[:, set_nodes['intervened']])
            x_gener_dict['children'].append(x_hat[:, set_nodes['children']])
            x_gener_dict['all'].append(x_hat)

        x_gener_dict_out = {}
        x_real_dict_out = {}
        for key, values in x_gener_dict.items():
            x_gener_dict_out[key] = torch.cat(values)
            if len(x_real_dict[key]) > 0:
                x_real_dict_out[key] = torch.cat(x_real_dict[key])

        data_loader.dataset.clean_intervention()

        return x_gener_dict_out, x_real_dict_out

    @torch.no_grad()
    def compute_counterfactual(self, batch, x_I, nodes_list, normalize,
                               return_type='sample'):
        z_factual, _ = self.model.encoder(batch.x, batch.edge_index, edge_attr=batch.edge_attr,
                                          return_mean=True, node_ids=batch.node_ids)

        # Encoder pass 2 CounterFactual
        z_cf_I, _ = self.model.encoder(batch.x_i, batch.edge_index_i, edge_attr=batch.edge_attr_i,
                                       return_mean=True, node_ids=batch.node_ids)

        z_factual = z_factual.reshape(batch.num_graphs, -1)
        z_cf_I = z_cf_I.reshape(batch.num_graphs, -1)

        # Replace z_cf of the intervened variables with z_cf_I
        z_dec = z_factual.clone()
        for node_name, _ in x_I.items():
            i = nodes_list.index(node_name)

            z_dec[:, self.z_dim * i:self.z_dim * (i + 1)] = z_cf_I[:, self.z_dim * i: self.z_dim * (i + 1)]

        z_dec = z_dec.reshape(-1, self.z_dim)

        x_CF, _ = self.model.decoder(z_dec, batch.edge_index_i, edge_attr=batch.edge_attr_i,
                                     return_type=return_type, node_ids=batch.node_ids)

        # Not normalized
        if normalize:
            x_CF = x_CF.view(batch.num_graphs, -1)
        else:
            x_CF = self.scaler.inverse_transform(x_CF.view(batch.num_graphs, -1))


        x_CF =  x_CF.view(batch.num_graphs, -1)
        z_cf_I = z_cf_I.reshape(batch.num_graphs, -1)
        z_dec = z_dec.reshape

        return x_CF, z_factual, z_cf_I, z_dec

    @torch.no_grad()
    def get_counterfactual_distr(self, data_loader,
                                 x_I=None,
                                 is_noise=False,
                                 return_z=False,
                                 num_batches=None,
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
        z_factual_dict = {'all': []}
        z_counterfactual_dict = {'all': []}

        for idx, batch in enumerate(iterator):
            batch = batch.to(self.device)
            if isinstance(num_batches, int) and idx > num_batches: break

            x_CF, z_factual, z_cf_I, z_dec = self.compute_counterfactual(batch=batch,
                                        x_I=data_loader.dataset.x_I,
                                        nodes_list=data_loader.dataset.nodes_list,
                                        normalize=normalize)

            z_factual_dict['all'].append(z_factual)
            z_counterfactual_dict['all'].append(z_cf_I.clone())

            u_factual = batch.u.view(batch.num_graphs, -1)

            x_cf_real, set_nodes = data_loader.dataset.get_counterfactual(
                x_factual=self.scaler.inverse_transform(self.get_x_graph(batch, 'x')),
                u_factual=u_factual,
                x_I=x_I,
                is_noise=is_noise,
                return_set_nodes=True)
            if x_cf_real is not None:
                if normalize:
                    x_cf_real = self.scaler.transform(x_cf_real)
                else:
                    x_cf_real = torch.Tensor(x_cf_real)

                x_real_dict['intervened'].append(x_cf_real[:, set_nodes['intervened']])
                x_real_dict['children'].append(x_cf_real[:, set_nodes['children']])
                x_real_dict['all'].append(x_cf_real)

            # Cf.shape [512,1] // CF.shape [1000, 3]
            x_gener_dict['intervened'].append(x_CF[:, set_nodes['intervened']])
            x_gener_dict['children'].append(x_CF[:, set_nodes['children']])
            x_gener_dict['all'].append(x_CF)

            if normalize:
                x_factual_dict['all'].append(self.get_x_graph(batch, 'x'))
            else:
                x_factual_dict['all'].append(self.scaler.inverse_transform(self.get_x_graph(batch, 'x')))

        x_gener_dict_out = {}
        x_real_dict_out = {}
        x_factual_dict_out = {}
        z_factual_dict_out = {}
        z_counterfactual_dict_out = {}
        for key, values in x_gener_dict.items():
            x_gener_dict_out[key] = torch.cat(values)
            if len(x_real_dict[key]) > 0:
                x_real_dict_out[key] = torch.cat(x_real_dict[key])

        for key, values in x_factual_dict.items():
            x_factual_dict_out[key] = torch.cat(values)
            z_factual_dict_out[key] = torch.cat(z_factual_dict[key])
            z_counterfactual_dict_out[key] = torch.cat(z_counterfactual_dict[key])

        data_loader.dataset.clean_intervention()

        if return_z:
            return x_gener_dict_out, z_counterfactual_dict_out, x_factual_dict_out, z_factual_dict_out
        else:
            return x_gener_dict_out, x_real_dict_out, x_factual_dict_out

    @torch.no_grad()
    def get_x(self, data_loader):
        iterator = iter(data_loader)
        self.eval()
        x = []
        for idx, batch in enumerate(iterator):
            x.append(self.get_x_graph(batch, 'x'))
        return torch.cat(x)

    @torch.no_grad()
    def get_reconstruction_distr(self, data_loader,
                                 num_batches=None,
                                 normalize=True):
        """
        Reconstruct all the features of all the  graphs in data loader, i.e.,
        Z \sim q(Z|X, A) and X_hat p(X | Z, A)
        Args:
            data_loader:
            num_batches:
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
        x, z = [], []
        x_real = []
        for idx, batch in enumerate(iterator):
            batch = batch.to(self.device)
            if isinstance(num_batches, int) and idx > num_batches: break
            z_hat, x_hat = self.model.reconstruct(batch)

            if normalize:
                x.append(x_hat.view(batch.num_graphs, -1))
                x_real.append(self.get_x_graph(batch, 'x'))

            else:
                x.append(self.scaler.inverse_transform(x_hat.view(batch.num_graphs, -1)))
                x_real.append(self.scaler.inverse_transform(self.get_x_graph(batch, 'x')))

            z.append(z_hat.reshape(batch.num_graphs, -1))

        return torch.cat(z), torch.cat(x), torch.cat(x_real)

    @torch.no_grad()
    def get_obs_distribution(self, data_loader):
        iterator = iter(data_loader)
        self.eval()
        x = []
        for idx, batch in enumerate(iterator):
            x.append(batch.x.view(batch.num_graphs, -1))

        return torch.cat(x)
