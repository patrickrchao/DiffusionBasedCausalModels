import os

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import FunctionTransformer
from torch.distributions import Laplace, Uniform, TransformedDistribution, SigmoidTransform

from utils.args_parser import mkdir
from utils.metrics.time import Timer
from utils.optimizers import get_optimizer, get_scheduler
from .nflib import AffineCL, NormalizingFlowModel, MLP1layer, MAF, NSF_AR, ARMLP, MLP4
import json


class CAREFL(pl.LightningModule):
    """
    Causal Autoregressive Flow
    """

    def __init__(self,
                 node_per_dimension_list,
                 distr_z='laplace',
                 flow_net_class='mlp',
                 flow_architecture='spline',
                 n_layers=1,
                 n_hidden=1,
                 parity=False,
                 scaler=None,
                 init=None):
        super(CAREFL, self).__init__()


        self.save_hyperparameters()
        self.total_dim = len(node_per_dimension_list)

        self.node_per_dimension = np.array(node_per_dimension_list)

        # prior
        if distr_z == 'laplace':
            prior = Laplace(torch.zeros(self.total_dim).to(self.device), torch.ones(self.total_dim).to(self.device))
        else:
            prior = TransformedDistribution(
                Uniform(torch.zeros(self.total_dim).to(self.device), torch.ones(self.total_dim).to(self.device)),
                SigmoidTransform().inv)
        # net type for flow parameters
        if flow_net_class == 'mlp':
            net_class = MLP1layer
        elif flow_net_class == 'mlp4':
            net_class = MLP4
        elif flow_net_class == 'armlp':
            net_class = ARMLP
        else:
            raise NotImplementedError('net_class {} not understood.'.format(self.config.flow.net_class))

        # flow type
        def ar_flow(hidden_dim):
            if flow_architecture in ['cl', 'realnvp']:
                return AffineCL(dim=self.total_dim, nh=hidden_dim, scale_base=self.config.flow.scale_base,
                                shift_base=self.config.flow.shift_base, net_class=net_class, parity=parity,
                                scale=self.config.flow.scale)
            elif flow_architecture == 'maf':
                return MAF(dim=self.total_dim, nh=hidden_dim, net_class=net_class, parity=parity)
            elif flow_architecture == 'spline':
                return NSF_AR(dim=self.total_dim, hidden_dim=hidden_dim, base_network=net_class)
            else:
                raise NotImplementedError('Architecture {} not understood.'.format(self.config.flow.architecture))

        flow_list = [ar_flow(n_hidden) for _ in range(n_layers)]

        self.flow_model = NormalizingFlowModel(prior, flow_list).to(self.device)

        if scaler is None:
            self.scaler = FunctionTransformer(func=lambda x: x,
                                              inverse_func=lambda x: x)
        else:
            self.scaler = scaler

        self.timer = Timer()

        if init == 'ortho':
            self.apply(init.init_weights_orthogonal)
        else:
            pass

        self.save_dir = None

        self.my_evaluator = None

    def get_x_graph(self, data, attr):
        x = getattr(data, attr)
        if attr in ['x', 'x_i']:
            x = x[data.mask]
        return x.view(data.num_graphs, -1)

    def set_my_evaluator(self, evaluator):
        self.my_evaluator = evaluator

    def monitor(self):
        return 'valid_objective'

    def monitor_mode(self):
        return 'max'

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
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        X = self.get_x_graph(batch, 'x')

        # compute loss
        _, prior_logprob, log_det = self.flow_model(X)
        objective = torch.mean(prior_logprob + log_det)

        self.log('train_objective', objective.item(), prog_bar=True)
        return -objective

    def on_train_epoch_start(self) -> None:
        self.timer.tic('train')

    def on_train_epoch_end(self, outputs) -> None:
        time = self.timer.toc('train')
        self.logger.experiment.add_scalar('train_time', time, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        batch = batch.to(self.device)
        X = self.get_x_graph(batch, 'x')  # shape [1000, 3] # works because 1 Dim X
        # compute loss

        _, prior_logprob, log_det = self.flow_model(X)
        objective = torch.mean(prior_logprob + log_det)

        self.log('valid_objective', objective.item(), prog_bar=True)

        return {f'valid_objective': objective.item()}

    def on_validation_epoch_start(self) -> None:
        self.timer.stop('train_total')

    def on_validation_epoch_end(self) -> None:
        self.timer.resume('train_total')

    def on_test_epoch_start(self) -> None:
        self.x_test = []
        self.x_hat = []
        return

    def test_step(self, batch, batch_idx):
        X = self.get_x_graph(batch, 'x')  # shape [1000, 3] # works because 1 Dim X
        # compute loss
        _, prior_logprob, log_det = self.flow_model(X)
        objective = torch.mean(prior_logprob + log_det)

        self.log('test_objective', objective.item(), prog_bar=True)

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

    def _forward_flow(self, data):

        return self.flow_model.forward(data.to(self.device))[0][-1].detach()

    def _backward_flow(self, latent):

        return self.flow_model.backward(latent.to(self.device))[0][-1].detach()

    @torch.no_grad()
    def evaluate(self, dataloader, name='test', save_dir=None, plots=False):
        self.my_evaluator.set_save_dir(save_dir if save_dir is not None else self.logger.save_dir, )
        self.my_evaluator.set_logger(self.logger)
        self.my_evaluator.set_current_epoch(100000)

        output = self.my_evaluator.evaluate(dataloader, name=name, plots=plots)
        return output

    @torch.no_grad()
    def get_observational_distr(self, data_loader,
                                use_links=True,
                                use_aggregated_posterior=False,
                                normalize=True):
        '''
        Get x generated distribution  w/o intervention or with diagonal adjacency.
        Parameters
        ----------
        use_links : bool
            If false, then uses an diagonal adjacency matrix to compute the distribution

        Returns
        -------
        z_list: torch.Tensor
            Latent code of the generated distribution
        x: torch.Tensor
            Generated distribution
        x_real: torch.Tensor
            distribution of the dataset (real data)
        '''

        iterator = iter(data_loader)
        self.eval()
        x, z_list = [], []
        x_real = []

        for idx, batch in enumerate(iterator):
            z = self.flow_model.prior.sample((batch.num_graphs,))
            x_hat = self._backward_flow(z)
            # x_real.append(self.scaler.inverse_transform(batch.x.view(batch.num_graphs, -1)))
            if normalize:
                x_real.append(self.get_x_graph(batch, 'x'))
                x.append(x_hat)
            else:
                x_real.append(self.scaler.inverse_transform(self.get_x_graph(batch, 'x')))
                x.append(self.scaler.inverse_transform(x_hat))

            z_list.append(z)

        data_loader.dataset.clean_intervention()

        return torch.cat(z_list), torch.cat(x), torch.cat(x_real)

    @torch.no_grad()
    def get_objective_metrics(self, data_loader, name):
        output = {}

        metrics = {'log_px': []}
        for idx, batch in enumerate(iter(data_loader)):
            X = self.get_x_graph(batch, 'x')  # shape [1000, 3] # works because 1 Dim X
            _, prior_logprob, log_det = self.flow_model(X)
            objective = torch.mean(prior_logprob + log_det)
            metrics['log_px'].append(objective.item())

        for key, value in metrics.items():
            output[f'{name}_{key}'] = np.mean(value)

        return output

    @torch.no_grad()
    def get_interventional_distr(self, data_loader,
                                 x_I,
                                 use_aggregated_posterior=False,
                                 normalize=True):
        '''
        Get x generated distribution  w/o intervention or with diagonal adjacency.
        Parameters
        ----------
        x_I : dict
             If x_I is None compute the distribution of the original SCM, if x_I is a dict
             then compute the interventional distribution. E.g. x_I = {'x1': 0} computes the
             interventional distribution with do(x1=0)
        Returns
        -------
        x_gener_dict_out: dict of torch.Tensor
            Generated distribution
        x_real_dict_out: dict of torch.Tensor
            distribution of the dataset (real data)
        '''
        assert isinstance(x_I, dict)
        assert len(x_I) == 1

        data_loader.dataset.set_intervention(x_I)
        dims_int = None
        assert len(data_loader.dataset.x_I) == 1
        for key, value in data_loader.dataset.x_I.items():
            dims_int = np.where(np.array(self.node_per_dimension) == key)[0]
        iterator = iter(data_loader)
        self.eval()
        x_gener_dict = {'intervened': [],
                        'children': [],
                        'all': []}
        x_real_dict = {'intervened': [],
                       'children': [],
                       'all': []}

        for idx, batch in enumerate(iterator):
            X = self.get_x_graph(batch, 'x')
            X_i = self.get_x_graph(batch, 'x_i')

            n_samples = X.shape[0]
            x_int = torch.zeros((1, self.total_dim))
            x_int[0, dims_int] = X_i[0, dims_int]  # Get the intervened normalized value

            z_int = self._forward_flow(x_int)[0, dims_int]
            # sample from prior and ensure z_intervention_index = z_int
            z = self.flow_model.prior.sample((n_samples,))
            z_est = torch.zeros((1, self.total_dim))
            z[:, dims_int] = z_est[:, dims_int] = z_int

            # propagate the latent sample through flow
            x_hat = self._backward_flow(z)

            if not normalize:
                x_hat = self.scaler.inverse_transform(x_hat)

            # Not normalized
            x_inter, set_nodes = data_loader.dataset.sample_intervention(x_I=x_I,
                                                                         n_samples=batch.num_graphs,
                                                                         return_set_nodes=True)
            if normalize:
                x_inter = self.scaler.transform(x_inter)
            else:
                x_inter = torch.tensor(x_inter)

            x_gener_dict['intervened'].append(x_hat[:, set_nodes['intervened']])
            x_gener_dict['children'].append(x_hat[:, set_nodes['children']])
            x_gener_dict['all'].append(x_hat)

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
    def compute_counterfactual(self, batch, x_I, z_I):

        for key, value in x_I.items():
            x_id = key
        X = self.get_x_graph(batch, 'x')
        X_i = self.get_x_graph(batch, 'x_i')
        # abduction:
        z_obs = self._forward_flow(X)
        z_cf_val = self._forward_flow(X_i)[:, x_id]
        z_obs[:, x_id] = z_cf_val
        # prediction (pass through the flow):
        x_CF = self._backward_flow(z_obs)

        return x_CF.view(batch.num_graphs, -1), z_obs.reshape(batch.num_graphs, -1), z_cf_val.reshape(
            batch.num_graphs, -1)

    @torch.no_grad()
    def get_counterfactual_distr(self, data_loader,
                                 x_I=None,
                                 is_noise=False,
                                 num_batches=None,
                                 normalize=True):
        assert isinstance(x_I, dict)
        assert len(x_I) == 1
        data_loader.dataset.set_intervention(x_I, is_noise=is_noise)

        for key, value in data_loader.dataset.x_I.items():
            dims_int = np.where(np.array(self.node_per_dimension) == key)[0]
        
        
        iterator = iter(data_loader)
        self.eval()

        x_gener_dict = {'intervened': [],
                        'children': [],
                        'all': []}
        x_real_dict = {'intervened': [],
                       'children': [],
                       'all': []}
        x_factual_dict = {'all': []}

        for idx, batch in enumerate(iterator):
            if isinstance(num_batches, int) and idx > num_batches: break
            X = self.get_x_graph(batch, 'x')
            X_i = self.get_x_graph(batch, 'x_i')
            # abduction:
            z_obs = self._forward_flow(X)
            
            z_cf_val = self._forward_flow(X_i)[:, dims_int]
            z_obs[:, dims_int] = z_cf_val
            # prediction (pass through the flow):
            if normalize:
                x_CF = self._backward_flow(z_obs)
            else:
                x_CF = self.scaler.inverse_transform(self._backward_flow(z_obs))

            # Not normalized
            x_cf_real, set_nodes = data_loader.dataset.get_counterfactual(
                x_factual=self.scaler.inverse_transform(self.get_x_graph(batch, 'x')),
                u_factual=batch.u.view(batch.num_graphs, -1),
                x_I=x_I,
                is_noise=is_noise,
                return_set_nodes=True)

            if normalize:
                x_cf_real = self.scaler.transform(x_cf_real)
            else:
                x_cf_real = torch.tensor(x_cf_real)
            # x_cf_real = self.scaler.transform(torch.tensor(x_cf_real))
            x_gener_dict['intervened'].append(x_CF[:, set_nodes['intervened']])
            x_gener_dict['children'].append(x_CF[:, set_nodes['children']])
            x_gener_dict['all'].append(x_CF)

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

    def get_data_is_toy(self):
        return True

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
        x, z = [], []
        x_real = []
        for idx, batch in enumerate(iterator):
            X = self.get_x_graph(batch, 'x')

            z_hat = self._forward_flow(X)
            # sample from prior and ensure z_intervention_index = z_int
            x_hat = self._backward_flow(z_hat)
            if normalize:
                x.append(x_hat)
                x_real.append(X)
            else:
                x.append(self.scaler.inverse_transform(x_hat))
                x_real.append(self.scaler.inverse_transform(X))

            z.append(z_hat)

        return torch.cat(z), torch.cat(x), torch.cat(x_real)
