import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from utils.args_parser import mkdir
from utils.constants import Cte
from utils.metrics.mmd import MMDLoss


class MyEvaluator:
    def __init__(self, model,
                 intervention_list,
                 scaler,
                 normalize=True):
        self.model = model
        self.logger = None
        self.mmd1 = MMDLoss(kernel_mul=2.0, kernel_num=5, num_samples=1000)
        self.save_dir = None
        self.intervention_list = intervention_list
        self.current_epoch = None
        self.scaler = scaler
        self.normalize = normalize

        return

    def set_model(self, model):
        self.model = model

    def set_logger(self, logger):
        self.logger = logger

    def set_save_dir(self, save_dir):
        self.save_dir = mkdir(save_dir)

    def set_current_epoch(self, current_epoch):
        self.current_epoch = current_epoch

    def logs_observation(self, x_obs, x_obs_gener, mode='train', name='observation', return_dict=False):
        output = {} if return_dict else None

        if len(x_obs) > 0:
            no_obs = True
        else:
            no_obs = False

        if no_obs:
            if not self.normalize:  # Data is not normalized
                max_ = x_obs.max(0)[0]
                max_[max_ == 0] = 1.
                x_obs = x_obs / max_
                x_obs_gener = x_obs_gener / max_
            # Added by Patrick Chao
            if x_obs.get_device()!= x_obs_gener.get_device():
                x_obs = x_obs.cpu()
                x_obs_gener = x_obs_gener.cpu()
            mse_mean = (x_obs.mean(0) - x_obs_gener.mean(0)) ** 2
            mse_std = (x_obs.std(0) - x_obs_gener.std(0)) ** 2
            diff_std = x_obs.std(0) - x_obs_gener.std(0)

            mmd1 = self.mmd1(x_obs, x_obs_gener)
            num_samples_2 = x_obs.shape[0] // 2
            mmd1_lb = self.mmd1(x_obs[:num_samples_2], x_obs[num_samples_2:])  # LB

            self.log_experiment_scalar(f'{mode}_{name}/mmd1', mmd1.item(), self.current_epoch, output=output)
            self.log_experiment_scalar(f'{mode}_{name}/mmd1_lb', mmd1_lb.item(), self.current_epoch, output=output)
            mse_samples = torch.norm(x_obs_gener - x_obs, p='fro', dim=-1) / x_obs_gener.shape[1]
            mse = mse_samples.mean()

            mse_samples_std = mse_samples.std()
            self.log_experiment_scalar(f'{mode}_{name}/x_mse', mse, self.current_epoch, output=output)
            self.log_experiment_scalar(f'{mode}_{name}/x_mse_std', mse_samples_std, self.current_epoch, output=output)

            mse = torch.sqrt(((x_obs_gener - x_obs) ** 2).mean(0))
            num_nodes = x_obs.shape[1]
            for i in range(num_nodes):
                self.log_experiment_scalar(f'{mode}_{name}/x{i + 1}_mse_mean', mse_mean[i], self.current_epoch,
                                           output=output)
                self.log_experiment_scalar(f'{mode}_{name}/x{i + 1}_mse_std', mse_std[i], self.current_epoch,
                                           output=output)
                self.log_experiment_scalar(f'{mode}_{name}/x{i + 1}_diff_std', diff_std[i], self.current_epoch,
                                           output=output)

                self.log_experiment_scalar(f'{mode}_{name}/x{i + 1}_mse', mse[i], self.current_epoch, output=output)

        if return_dict:
            return output

    def logs_observation_reduced(self, x_obs, x_obs_gener, mode='train', name='observation', return_dict=False):
        output = {} if return_dict else None

        mmd1 = self.mmd1(x_obs, x_obs_gener)

        self.log_experiment_scalar(f'{mode}_{name}/mmd1', mmd1.item(), self.current_epoch, output=output)

        mse = torch.norm(x_obs_gener - x_obs, p='fro', dim=-1).mean() / x_obs_gener.shape[1]
        self.log_experiment_scalar(f'{mode}_{name}/x_mse', mse, self.current_epoch, output=output)

        if return_dict:
            return output

    def log_experiment_scalar(self, name, value, epoch, output=None):
        if isinstance(output, dict):
            output[name] = float(value)
        else:
            self.logger.experiment.add_scalar(name, value, epoch)

    @torch.no_grad()
    def complete_logs(self, data_loader, name, plots=False):
        # IID observations
        _, x_diag, _ = self.model.get_observational_distr(data_loader, use_links=False)

        z_hat, x_hat, x_real = self.model.get_observational_distr(data_loader)

        mmd1 = self.mmd1(x_real, x_diag)

        self.logger.experiment.add_scalar(f'{name}_observation/mmd1_ub', mmd1.item(), self.current_epoch)

        self.logs_observation(x_obs=x_real, x_obs_gener=x_hat, mode=name, name='observation')
        if plots: self.plot_obs(x_hat, x_real=x_real, label='x_obs_gener/', iter=self.current_epoch)

        z_hat, x_hat, x_real = self.model.get_observational_distr(data_loader, use_aggregated_posterior=True)
        self.logs_observation_reduced(x_obs=x_real, x_obs_gener=x_hat, mode=name, name='observation_aggr')

        z, x_recons, x_r_real = self.model.get_reconstruction_distr(data_loader)
        self.logs_observation(x_obs=x_r_real, x_obs_gener=x_recons, mode=name, name='recons')

        if plots: self.plot_obs(x_recons, x_real=x_r_real, label='x_obs_recons/', iter=self.current_epoch)

        for x_I, i_label in self.intervention_list:
            X_gener_dict, X_real_dict = self.model.get_interventional_distr(data_loader, x_I=x_I)

            label = '_'.join([f'{k}_{i_label}' for k, v in x_I.items()])

            for key, value in X_gener_dict.items():
                if key in X_real_dict:
                    self.logs_observation(x_obs=X_real_dict[key], x_obs_gener=value, mode=name,
                                          name=f'inter_{label}_{key}')
                else:  # no ground truth
                    pass
                    # self.logs_observation(x_obs=X_real_dict, x_obs_gener=value, mode=name,
                    #                       name=f'inter_{label}_{key}')

            X_gener_dict, X_real_dict, _ = self.model.get_counterfactual_distr(data_loader, x_I=x_I, is_noise=True)

            for key, value in X_gener_dict.items():
                if key in X_real_dict:
                    self.logs_observation(x_obs=X_real_dict[key], x_obs_gener=value, mode='test',
                                          name=f'cf_{label}_{key}_noise')

                    if plots: self.plot_obs(value, x_real=X_real_dict[key], label=f'cf_{label}_noise_gener/',
                                            iter=self.current_epoch)

            X_gener_dict, X_real_dict, _ = self.model.get_counterfactual_distr(data_loader, x_I=x_I, is_noise=False)

            for key, value in X_gener_dict.items():
                if key in X_real_dict:
                    self.logs_observation(x_obs=X_real_dict[key], x_obs_gener=value, mode='test',
                                          name=f'cf_{label}_{key}')

                    if plots: self.plot_obs(value, x_real=X_real_dict[key], label=f'cf_{label}_gener/',
                                            iter=self.current_epoch)

    @torch.no_grad()
    def evaluate(self, dataloader, name='test', plots=False):

        output = {}
        dataset = dataloader.dataset

        o = self.model.get_objective_metrics(dataloader, name)

        output.update(o)

        # IID observations
        _, x_diag, x_real = self.model.get_observational_distr(dataloader,
                                                               use_links=False,
                                                               normalize=self.normalize)
        o = self.logs_observation(x_obs=x_real,
                                  x_obs_gener=x_diag,
                                  mode=name,
                                  name='observation_independent',
                                  return_dict=True)

        if plots:
            self.plot_obs(x_diag, x_real=x_real,
                          label=f'{name}_observation_independent/', iter=100000)
        output.update(o)

        # SCM observations
        z_hat, x_hat, x_real = self.model.get_observational_distr(dataloader,
                                                                  normalize=self.normalize)
        mmd1 = self.mmd1(x_real, x_diag)
        output[f'{name}_observation/mmd1_ub'] = mmd1.item()

        o = self.logs_observation(x_obs=x_real, x_obs_gener=x_hat, mode=name, name='observation',
                                  return_dict=True)
        if plots:
            self.plot_obs(x_hat, x_real=x_real,
                          label=f'{name}_observation/', iter=100000)
        output.update(o)

        # Obs with aggregated posterior
        z_hat, x_hat, x_real = self.model.get_observational_distr(dataloader,
                                                                  use_aggregated_posterior=True,
                                                                  normalize=self.normalize)
        o = self.logs_observation_reduced(x_obs=x_real, x_obs_gener=x_hat, mode=name, name='observation_aggr',
                                          return_dict=True)
        if plots:
            self.plot_obs(x_hat, x_real=x_real,
                          label=f'{name}_observation_aggr/', iter=100000)
        output.update(o)
        # Reconstruction
        z, x_recons, x_r_real = self.model.get_reconstruction_distr(dataloader,
                                                                    normalize=self.normalize)
        o = self.logs_observation(x_obs=x_r_real, x_obs_gener=x_recons, mode=name, name='recons',
                                  return_dict=True)
        if plots:
            self.plot_obs(x_recons, x_real=x_r_real,
                          label=f'{name}_reconstruction/', iter=100000)
        output.update(o)

        # Interventions and CFs
        for x_I, i_label in self.intervention_list:
            X_gener_dict, X_real_dict = self.model.get_interventional_distr(dataloader,
                                                                            x_I=x_I,
                                                                            normalize=self.normalize)

            label = '_'.join([f'{k}_{i_label}' for k, v in x_I.items()])

            if plots:
                if dataset.is_toy():
                    self.plot_obs(X_gener_dict['all'], x_real=X_real_dict['all'],
                                  label=f'{name}_intervention_{label}_gener/', iter=100000)
                else:
                    self.plot_obs(X_gener_dict['all'], label=f'{name}_intervention_{label}_gener/', iter=100000)

            data_is_toy = False
            for key, value in X_gener_dict.items():
                if key in X_real_dict:
                    data_is_toy = True
                    o = self.logs_observation(x_obs=X_real_dict[key], x_obs_gener=value, mode=name,
                                              name=f'inter_{label}_{key}',
                                              return_dict=True)
                else:
                    o = self.logs_observation(x_obs=X_real_dict, x_obs_gener=value, mode=name,
                                              name=f'inter_{label}_{key}',
                                              return_dict=True)
                output.update(o)

            X_gener_dict, X_real_dict, X_factual = self.model.get_counterfactual_distr(dataloader,
                                                                                       x_I=x_I,
                                                                                       is_noise=True,
                                                                                       normalize=self.normalize)
            if data_is_toy:

                if plots:
                    self.plot_obs(X_gener_dict['all'], x_real=X_real_dict['all'],
                                  label=f'{name}_cf_gener/', iter=100000)

                for key, value in X_gener_dict.items():
                    o = self.logs_observation(x_obs=X_real_dict[key], x_obs_gener=value, mode=name,
                                              name=f'cf_{label}_{key}_noise',
                                              return_dict=True)

                    output.update(o)

            if plots:
                if data_is_toy:
                    self.plot_obs(X_gener_dict['all'], x_real=X_real_dict['all'],
                                  label=f'{name}_cf_gener/', iter=100000)
                else:
                    self.plot_obs(X_gener_dict['all'], label=f'{name}_cf_gener/', iter=100000)

            for key, value in X_gener_dict.items():
                if data_is_toy:
                    o = self.logs_observation(x_obs=X_real_dict[key], x_obs_gener=value, mode=name,
                                              name=f'cf_{label}_{key}',
                                              return_dict=True)
                else:
                    o = self.logs_observation(x_obs=X_real_dict, x_obs_gener=value, mode=name,
                                              name=f'cf_{label}_{key}',
                                              return_dict=True)
                output.update(o)

        for key, value in output.items():
            print(f"{key}: {value}")
        return output

    def evaluate_cf_fairness(self, data_module):
        # Mask
        attributes_dict = data_module.get_attributes_dict()

        mask_unaware = attributes_dict['fair_attributes'] \
                       + attributes_dict['unfair_attributes']

        mask_fair = attributes_dict['fair_attributes']

        if len(mask_fair) > 0:
            fair_available = True
        else:
            fair_available = False

        x_cf_0_dict, z_cf_0_dict, x_f_dict, z_f_dict = self.model.get_counterfactual_distr(
            data_loader=data_module.test_dataloader(),
            x_I={Cte.SENS: 0},
            is_noise=False, return_z=True)
        x_cf_1_dict, z_cf_1_dict, _, _ = self.model.get_counterfactual_distr(data_loader=data_module.test_dataloader(),
                                                                             x_I={Cte.SENS: 1},
                                                                             is_noise=False, return_z=True)
        # xf0 = x_f_dict_0['all']
        # xf1 = x_f_dict_1['all']
        #
        # zf0 = z_f_dict_0['all']
        # zf1 = z_f_dict_1['all']
        #
        # print('x', mean_squared_error(xf0, xf1))
        # print('z', mean_squared_error(zf0, zf1))

        # normalized data gen
        x_cf_0 = x_cf_0_dict['all']
        x_cf_1 = x_cf_1_dict['all']
        x_f = x_f_dict['all']

        z_cf_0 = z_cf_0_dict['all']
        z_cf_1 = z_cf_1_dict['all']
        z_f = z_f_dict['all']
        #
        mask_0 = [x[0] for x in (x_f[:, 0] != 1).nonzero().tolist()]
        mask_1 = [x[0] for x in (x_f[:, 0] == 1).nonzero().tolist()]

        # x_cf_0: do(a=0) should be for all a=1
        x_cf_1[mask_1, :] = x_cf_0[mask_1, :]
        x_cf = x_cf_1.clone()

        z_cf_1[mask_1, :] = z_cf_0[mask_1, :]
        z_cf = z_cf_1.clone()

        dict_datasets = {}
        dict_datasets['full'] = {}
        dict_datasets['full']['X_train'] = data_module.get_normalized_X(mode='train')
        dict_datasets['full']['Y_train'] = data_module.train_dataset.Y.ravel()
        dict_datasets['full']['X_test'] = data_module.get_normalized_X(mode='test')
        dict_datasets['full']['Y_test'] = data_module.test_dataset.Y.ravel()
        dict_datasets['full']['X_cf'] = x_cf
        dict_datasets['full']['X_f'] = x_f

        dict_datasets['unaware'] = {}
        dict_datasets['unaware']['X_train'] = data_module.get_normalized_X(mode='train')[:, mask_unaware]
        dict_datasets['unaware']['Y_train'] = data_module.train_dataset.Y.ravel()
        dict_datasets['unaware']['X_test'] = data_module.get_normalized_X(mode='test')[:, mask_unaware]
        dict_datasets['unaware']['Y_test'] = data_module.test_dataset.Y.ravel()
        dict_datasets['unaware']['X_cf'] = x_cf[:, mask_unaware]
        dict_datasets['unaware']['X_f'] = x_f[:, mask_unaware]

        if fair_available:
            dict_datasets['fair'] = {}
            dict_datasets['fair']['X_train'] = data_module.get_normalized_X(mode='train')[:, mask_fair]
            dict_datasets['fair']['Y_train'] = data_module.train_dataset.Y.ravel()
            dict_datasets['fair']['X_test'] = data_module.get_normalized_X(mode='test')[:, mask_fair]
            dict_datasets['fair']['Y_test'] = data_module.test_dataset.Y.ravel()
            dict_datasets['fair']['X_cf'] = x_cf[:, mask_fair]
            dict_datasets['fair']['X_f'] = x_f[:, mask_fair]

        # get Z_train
        data_module.set_shuffle_train(False)
        z_train, _, _ = self.model.get_reconstruction_distr(data_module.train_dataloader())
        z_test, _, _ = self.model.get_reconstruction_distr(data_module.test_dataloader())

        dict_datasets['VACA'] = {}  # note: X here is is Z
        dict_datasets['VACA']['X_train'] = z_train[:, self.model.z_dim:]
        dict_datasets['VACA']['Y_train'] = data_module.train_dataset.Y.ravel()
        dict_datasets['VACA']['X_test'] = z_test[:, self.model.z_dim:]
        dict_datasets['VACA']['Y_test'] = data_module.test_dataset.Y.ravel()
        dict_datasets['VACA']['X_cf'] = z_cf[:, self.model.z_dim:]
        dict_datasets['VACA']['X_f'] = z_f[:, self.model.z_dim:]

        output = {}

        for dataset_name, XY_dict in dict_datasets.items():
            score_lr = []
            score_svm = []
            unfairness_lr = []
            unfairness_svm = []
            for seed in range(1, 11):
                def get_logistic_regression():
                    return LogisticRegression(class_weight='balanced', random_state=seed)

                def get_support_vector_machine():
                    return SVC(class_weight='balanced', probability=True, random_state=seed)

                # def get_decision_tree():
                #     return DecisionTreeClassifier(class_weight='balanced', random_state=seed, criterion='entropy',
                #                                   max_depth=2)

                dict_clf_generator = {'lr': get_logistic_regression,
                                      'svm': get_support_vector_machine}
                # 'dt': get_decision_tree}

                for clf_name, clf_generator in dict_clf_generator.items():
                    clf = clf_generator()

                    clf.fit(XY_dict['X_train'], XY_dict['Y_train'])

                    y_pred_f = clf.predict(XY_dict['X_test'])

                    # score_a = accuracy_score(XY_dict['Y_test'], y_pred_f)
                    score_f1 = f1_score(XY_dict['Y_test'], y_pred_f)

                    if clf_name == 'lr':
                        score_lr.append(score_f1)
                        # print('clf_name', clf_name, 'seed', seed, 'score_f1', score_f1)
                    else:
                        score_svm.append(score_f1)

                    p_pred_f = clf.predict_proba(XY_dict['X_test'])
                    p_pred_cf = clf.predict_proba(XY_dict['X_cf'])

                    unfairness_p = ((abs(p_pred_f[:, 1] - p_pred_cf[:, 1]))).mean()
                    if clf_name == 'lr':
                        unfairness_lr.append(unfairness_p)
                    else:
                        unfairness_svm.append(unfairness_p)

                    # 1: male 700/1000 samples are male
                    # 0: female 300/1000 samples are female

            for clf_name, clf_generator in dict_clf_generator.items():
                if clf_name == 'lr':
                    score = score_lr.copy()
                    unfairness = unfairness_lr.copy()
                else:
                    score = score_svm.copy()
                    unfairness = unfairness_svm.copy()
                print(f'{dataset_name} : {clf_name}')
                print(f'\t f1: {round(np.mean(score) * 100, 2)}, +- {round(np.std(score) * 100, 2)}')
                output[f'{dataset_name}_{clf_name}_f1_mean'] = round(np.mean(score) * 100, 2)
                output[f'{dataset_name}_{clf_name}_f1_std'] = round(np.std(score) * 100, 2)
                # print(f'\t acc: {score_a * 100}, std')
                print(
                    f'\t unfairness prob all: {round(np.mean(unfairness) * 100, 2)}, +- {round(np.std(unfairness) * 100, 2)}')
                output[f'{dataset_name}_{clf_name}_unfairness_mean'] = round(np.mean(unfairness) * 100, 2)
                output[f'{dataset_name}_{clf_name}_unfairness_std'] = round(np.std(unfairness) * 100, 2)

        return output

    def plot_densities(self, y0, y1, name, inter):
        fig, ax = plt.subplots()
        sns.distplot(y0, ax=ax, kde=True, color='blue')
        sns.distplot(y1, ax=ax, kde=True, color='orange')
        self.save_fig(f"{name}_{inter}", fig, global_step=0)
        matplotlib.pyplot.close('all')

    def plot_obs(self, x, x_real=None, label='x_obs/', iter=0):

        label = label.replace('/', '')

        columns = [f"dim_{i + 1}" for i in range(x.shape[1])]
        if isinstance(x, torch.Tensor): x = x.numpy()
        if x_real is None:
            df = pd.DataFrame(data=x, columns=columns)
            fig = sns.pairplot(df)
        else:
            if isinstance(x_real, torch.Tensor): x_real = x_real.numpy()
            num1 = x.shape[0]
            x_total = np.concatenate([x, x_real], 0)
            df = pd.DataFrame(data=x_total, columns=columns)
            df['Distribution'] = 'Real'
            df['Distribution'].iloc[:num1] = 'Gener'
            fig = sns.pairplot(df, hue='Distribution',
                               plot_kws={'alpha': 0.3}, hue_order=['Real', 'Gener'],
                               diag_kind="hist")

        self.save_fig(label, fig, iter)
        matplotlib.pyplot.close('all')

        return

    def save_fig(self, name, fig, global_step):

        img_folder = os.path.join(self.save_dir, 'images')

        fig.savefig(os.path.join(img_folder, f'{name}_{global_step}.png'))

    def save_grid(self, name, grid, global_step):
        img_folder = os.path.join(self.save_dir, 'images')

        matplotlib.image.imsave(os.path.join(img_folder, f'{name}_{global_step}.png'),
                                grid.transpose(2, 0).cpu().numpy())
