import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway
from scipy.stats import ttest_ind

from utils.args_parser import mkdir
from utils.constants import Cte


class ResultsManager:
    def __init__(self, root_dir, objective_mmd=False):
        self.root_dir = root_dir

        self.save_dir = os.path.join('images', root_dir.replace(os.sep, '_'))
        mkdir(self.save_dir)

        df_all = create_df_results(root_dir, add_time=True)
        df_all = df_all.rename(columns={c: c.replace(os.sep, '_') for c in df_all.columns})

        df_all['model_params_num_hidden_dec'] = df_all['model_params_h_dim_list_dec'].apply(get_number_hidden_layers)
        df_all['model_params_num_hidden_enc'] = df_all['model_params_h_dim_list_enc'].apply(get_number_hidden_layers)

        df_all['model_params_h_dim_list_enc'] = df_all['model_params_h_dim_list_enc'].apply(process_list_params)
        df_all['model_params_h_dim_list_dec'] = df_all['model_params_h_dim_list_dec'].apply(process_list_params)
        df_all['model_params_h_dim_list'] = 'dec_' + df_all['model_params_h_dim_list_dec'] + '_enc_' + df_all[
            'model_params_h_dim_list_enc']

        df_all.rename(columns={"model_name": "Model",
                               "dataset_name": "Dataset",
                               "dataset_params_equations_type": "SEM"}, inplace=True)

        print(f"Number of experiments: {len(df_all)}")
        print(f"Datasets: {df_all['Dataset'].unique()}")
        print(f"Models: {df_all['Model'].unique()}")
        print(f"Architectures: {df_all['model_params_architecture'].unique()}")

        columns_list = list(df_all.columns)
        self.columns_list = columns_list

        df_all['Model'] = df_all['Model'].replace({'mcvae': 'MultiCVAE',
                                                   'vcause_piwae': 'VACA',
                                                   'vcause': 'VACA',
                                                   'carefl': 'CAREFL'}
                                                  )

        metrics_dict = {'IWAE 100': ['test_iwae_100'],
                        'MMD Obs.': ['test_observation_mmd1'],
                        'MMD Inter.': get_elements(columns_list, ['test', 'mmd', 'inter', 'children'], ['mmd1_lb']),
                        'MeanE.': get_elements(columns_list, ['test', 'mse_mean', '_inter_', 'children']),
                        'StdE.': get_elements(columns_list, ['test', 'mse_std', 'inter', 'children']),
                        'MSE CF': get_elements(columns_list, ['test', '_cf_', 'x_mse', 'children', 'noise'],
                                               ['std', 'x1', 'age']),
                        'SSE CF': get_elements(columns_list,
                                               ['test', '_cf_', 'x_mse_std', 'children', 'noise', 'x1', 'age']),
                        'MRE CF N': get_elements(columns_list, ['test', '_cf_', 'x_mse', 'children'],
                                                 ['std', 'noise', 'x1', 'age']),
                        'SDRE CF N': get_elements(columns_list, ['test', '_cf_', 'x_mse_std', 'children'],
                                                  ['noise', 'x1', 'age'])
                        }
        self.metrics_dict = metrics_dict

        for key, values in metrics_dict.items():
            if key in ['test_iwae_100', 'test_observation_mmd1']:
                assert len(values) == 1
                df_all[key] = df_all[values[0]]
                continue
            print(key)
            print_cols(values)
            df_all[key] = df_all[values].mean(1)

        self.df = df_all

        self.df_best = None

        # Hyperparameters cross-validated

        self.cv_dict = {'CAREFL': ['model_params_n_layers',
                                   'model_params_n_hidden'],
                        'VACA': ['dataset_params_likelihood_names',
                                 'model_params_h_dim_list',
                                 'model_params_dropout_adj_pa_rate',
                                 'model_params_dropout_adj_pa_prob_keep_self',
                                 'model_params_residual'
                                 ],
                        'MultiCVAE': ['model_params_z_dim',
                                      'dataset_params_lambda_',
                                      'model_params_h_dim_list',
                                      'model_params_drop_rate',
                                      ]}

        # Objective metrics for each model
        if not objective_mmd:
            self.objective_metric = {'CAREFL': 'valid_log_px',
                                     'VACA': 'valid_iwae_100',
                                     'MultiCVAE': 'valid_iwae_100'}
        else:
            self.objective_metric = {'CAREFL': 'MMD Obs.',
                                     'VACA': 'MMD Obs.',
                                     'MultiCVAE': 'MMD Obs.'}

        # Minimun number of hidden layers in the decoder (model_params_num_hidden_dec) per dataset

        self.min_h_layers = {Cte.TRIANGLE: 1,
                             Cte.CHAIN: 1,
                             Cte.LOAN: 2,
                             Cte.COLLIDER: 0,
                             Cte.MGRAPH: 0,
                             Cte.ADULT: 2
                             }

        self.dataset_name_order = ['collider', 'mgraph', 'triangle', 'chain', 'loan', 'adult']
        self.sem_name_order = ['linear', 'non-linear', 'non-additive']
        self.model_name_order = ['MultiCVAE', 'CAREFL', 'VACA']

    def filter_valid_configurations(self, df):
        cond = df['Model'] != 'VACA'
        for dataset, min_h_layers in self.min_h_layers.items():
            cond_i = (df.model_params_num_hidden_dec >= min_h_layers) & (df.Dataset == dataset)
            cond = cond | cond_i
        return df[cond]

    def load_df_best(self, safe=0, dim_z=4):
        '''
        we need dimension z to remove those experiments that we use for the experiments on cross validating dim(z)
        '''
        print('\n\nComputing best configurations for each model and SEM:')
        cols = ['Model', 'Dataset', 'SEM', 'json_filename', 'num_parameters']
        cols.extend(get_elements(self.columns_list, ['dataset_params']))
        cols.extend(get_elements(self.columns_list, ['model_params']))
        metrics_cols = list(set(list(self.objective_metric.values())))
        cols.extend(metrics_cols)
        cols.extend(list(self.metrics_dict.keys()))
        cols = list(set(cols))
        df = self.df.copy()[cols]

        df = self.filter_valid_configurations(df)

        best_models_file = os.path.join(self.save_dir, 'best_models.txt')

        best_models_list = []
        for dataset_name, df_dataset in df.groupby('Dataset'):
            for m_name, df_m in df_dataset.groupby('Model'):
                print('--------')
                if m_name == 'VACA':
                    df_m = df_m[df_m.model_params_z_dim == dim_z]
                for d_name, df_md in df_m.groupby('SEM'):
                    print(f'{dataset_name} : {m_name} : {d_name}')

                    with open(best_models_file, 'a') as f:
                        f.write(f'{dataset_name} : {m_name} : {d_name}\n')
                    df_md_g = df_md.groupby(self.cv_dict[m_name], dropna=False).agg(['mean', 'std', 'count'])[
                        self.objective_metric[m_name]]

                    if safe > 0:
                        for best_config, df_best_config in df_md_g.sort_values(
                                by='mean').iterrows():
                            print(f"len: {df_best_config['count']}")
                            if df_best_config['count'] >= (safe - 1):
                                break
                    else:
                        best_config = df_md_g['mean'].idxmax()

                    df_best_md = df_md.copy()

                    for k, v in zip(self.cv_dict[m_name], best_config):
                        with open(best_models_file, 'a') as f:
                            f.write(f'\t{k}: {v}\n')
                        print(f'\t{k}: {v}')
                        df_best_md = df_best_md[df_best_md[k] == v]

                    print(f"Num of entries: {len(df_best_md)}")
                    with open(best_models_file, 'a') as f:
                        best = df_best_md.loc[df_best_md[self.objective_metric[m_name]].idxmax()]
                        f.write(f"\t{best['json_filename']}\n")
                        f.write(f"\tnum_parameters: {best['num_parameters']}\n")
                    print(df_best_md.loc[df_best_md[self.objective_metric[m_name]].idxmax()]['json_filename'])
                    get_unique_parameteres(self.columns_list,
                                           df_i=df_best_md,
                                           type_list=['model'])

                    my_mean, my_std, _ = df_md_g.loc[best_config]
                    print(f"{self.objective_metric[m_name]}: {my_mean:.3f} +- {my_std:.3f}\n")
                    if safe > 0: assert len(df_best_md) >= (
                            safe - 1), f'Number of elements different from number of seeds {len(df_best_md)}'
                    best_models_list.append(df_best_md)

        df_best = pd.concat(best_models_list)

        print('\n\nModels we are comparing:')

        for m in df_best['Model'].unique():
            print(f"\t{m}")

        self.df_best = df_best

    def generate_latex_table_comparison(self, metrics_to_plot=None,
                                        include_num_params=True):
        # Table 2 in the paper
        if not isinstance(metrics_to_plot, list):
            metrics_to_plot = [1, 2, 3, 4, 7, 8]
        cols_metrics = list(self.metrics_dict.keys())
        if include_num_params:
            cols_metrics.append('Num. parameters')
            metrics_to_plot.append(9)
        for i, c in enumerate(cols_metrics):
            add = 'True' if i in metrics_to_plot else 'False'
            print(f"({i}) [{add}] {c}")

        df_latex = self.df_best.copy()

        group_by_columns = ['Dataset', 'SEM', 'Model']

        dataset_dict = {'collider': 0,
                        'triangle': 1,
                        'loan': 2,
                        'm_graph': 3,
                        'chain': 4,
                        Cte.ADULT: 5}

        sem_dict = {'linear': 0,
                    'non-linear': 1,
                    'non-additive': 2
                    }

        model_dict = {'MultiCVAE': 0,
                      'CAREFL': 1,
                      'VACA': 2
                      }
        df_latex['Dataset'] = df_latex['Dataset'].replace(dataset_dict)
        df_latex['Model'] = df_latex['Model'].replace(model_dict)

        df_latex['SEM'] = df_latex['SEM'].replace(sem_dict)
        if include_num_params:
            df_latex['Num. parameters'] = df_latex['num_parameters']

        print(f"Number of elements to create the table: {len(df_latex)}")

        df_mean = df_latex.groupby(group_by_columns).mean()[cols_metrics] * 100
        if include_num_params:
            df_mean['Num. parameters'] = df_mean['Num. parameters'] / 100
        df_mean = df_mean.rename(index={v: k for k, v in dataset_dict.items()},
                                 level=0).rename(index={v: k for k, v in sem_dict.items()},
                                                 level=1).rename(index={v: k for k, v in model_dict.items()},
                                                                 level=2).applymap(lambda x: '{0:.2f}'.format(x))
        df_std = df_latex.groupby(group_by_columns).std()[cols_metrics] * 100
        if include_num_params:
            df_std['Num. parameters'] = df_std['Num. parameters'] / 100
        df_std = df_std.rename(index={v: k for k, v in dataset_dict.items()},
                               level=0).rename(index={v: k for k, v in sem_dict.items()},
                                               level=1).rename(index={v: k for k, v in model_dict.items()},
                                                               level=2).applymap(lambda x: '{0:.2f}'.format(x))

        df_comparison = df_mean + '$\pm$' + df_std
        table_file = os.path.join(self.save_dir, f'my_table_all.tex')
        with open(table_file, 'w') as tf:
            tf.write(df_comparison.iloc[:, metrics_to_plot].to_latex(escape=False))

        return df_comparison

    def generate_latex_table_propositions(self):
        raise NotImplementedError

    def budget(self, only_valid=True, filter_and=None):
        print('\nComputing budget')

        df = self.df.copy()
        if only_valid:
            df = self.filter_valid_configurations(df)

        if isinstance(filter_and, dict):
            cond = df['Model'] == 'VACA'
            for col, values in filter_and.items():
                cond_i = df[col].isin(values)
                cond = cond & cond_i

            cond = cond | (df['Model'] != 'VACA')
            df = df[cond]

        groupby = ['Dataset', 'SEM', 'Model']

        print(df.groupby(groupby).count()['json_filename'])

    def time_complexity(self, n=None,
                        replace=False,
                        time_list=None,
                        max_num_parameters=None,
                        ylim=None,
                        font_scale=1):
        df = self.df.copy()
        if time_list is None:
            train_time_str = 'Total training time (min)'
            train_col = 'train_time_total'
        else:
            train_time_str = time_list[0]
            train_col = time_list[1]
        num_params_str = 'Num. parameters'
        groupby_cols = ['Model', 'Dataset']
        metrics_cols = [train_time_str, num_params_str, 'model_params_num_hidden_dec']
        cols_time = [*groupby_cols, *metrics_cols, 'train_epochs']
        # cond_1 = (df['model_params_z_dim'] == 4) & (df['Model'] == 'VACA')
        # cond_2 = df['Model'] == 'MultiCVAE'
        # cond_3 = df['Model'] == 'CAREFL'
        # cond = cond_1 | cond_2 | cond_3
        # df = df[cond]

        cond = (df.model_params_num_hidden_dec > 0) | (df.model_params_num_hidden_dec == -1)
        df_time = df[cond]
        if isinstance(max_num_parameters, int):
            df_time = df_time[df_time.num_parameters < max_num_parameters]

        df_time = self.order_by_model(df_time)
        df_time = df_time.rename(columns={train_col: train_time_str})
        df_time = df_time.rename(columns={"num_parameters": num_params_str})[cols_time]
        df_time[train_time_str] = df_time[train_time_str] / 60

        print(df_time.groupby(groupby_cols).agg(['mean', 'std', 'max'])[metrics_cols])
        print(f'\nHow many experiments have we run for each model and dataset?')

        for (m_name, d_name), df_g in df_time.groupby(groupby_cols):
            print(f"{m_name} {d_name}: {len(df_g)}")

        print('\nPlotting training time for the three different models')
        plt.close('all')

        ax = sns.boxplot(x="Model", y=train_time_str, data=df_time)
        ax.set(ylim=ylim)
        plt.show()

        ax.get_figure().savefig(os.path.join(self.save_dir, 'time_complexity_all.png'))

        g = sns.catplot(x="Model", y=train_time_str, data=df_time, showfliers=False,
                        kind="box", legend=True,
                        hue='Dataset'
                        )
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        g.set_xlabels('')
        plt.show()
        g.savefig(os.path.join(self.save_dir, f'time_complexity_all_per_dataset.png'))

        print(df_time['Dataset'].unique())

        df_time = df_time.rename(columns={'train_epochs': 'Num. Epochs'})

        df_time = self.order_by_dataset(df_time)
        g = sns.catplot(x="Model", y='Num. Epochs', data=df_time, showfliers=True,
                        kind="box", legend=False,
                        hue='Dataset'
                        )
        plt.legend(loc='best')
        g.set_xlabels('')
        # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        g.savefig(os.path.join(self.save_dir, f'time_complexity_all_epochs_per_dataset.png'))

        print(df_time.groupby(['Model']).agg(['mean', 'std'])[train_time_str])
        print(f'\nAre the training times significantly different?')
        if n is not None:
            time_carefl = df_time[df_time.Model == 'CAREFL'][train_time_str].sample(n, replace=replace)
            time_vcause = df_time[df_time.Model == 'VACA'][train_time_str].sample(n, replace=replace)
            time_multicvae = df_time[df_time.Model == 'MultiCVAE'][train_time_str].sample(n, replace=replace)
        else:
            time_carefl = df_time[df_time.Model == 'CAREFL'][train_time_str]
            time_vcause = df_time[df_time.Model == 'VACA'][train_time_str]
            time_multicvae = df_time[df_time.Model == 'MultiCVAE'][train_time_str]

        statistic, pvalue = ttest_ind(time_vcause, time_carefl)

        print(f'p-value of the T-test for VACA and CAREFL: {pvalue:.4f}')

        statistic, pvalue = ttest_ind(time_multicvae, time_carefl)
        print(f'p-value of the T-test for CAREFL and MultiCVAE: {pvalue:.4f}')

        statistic, pvalue = ttest_ind(time_multicvae, time_vcause)
        print(f'p-value of the T-test for VACA and MultiCVAE: {pvalue:.4f}')

        statistic, pvalue = f_oneway(list(time_carefl.values),
                                     list(time_multicvae.values),
                                     list(time_vcause.values))
        print(f'p-value of the f_oneway for : {pvalue:.4f}')

        print(f'\nAre the training times significantly different PER DATASET?')
        if font_scale != 1:
            sns.set(font_scale=font_scale)
            sns.set_style("white")
        for d_name, df_data in df_time.groupby(['Dataset']):
            print(f'\nDataset: {d_name}')
            time_carefl = df_data[df_data.Model == 'CAREFL'][train_time_str]
            time_vcause = df_data[df_data.Model == 'VACA'][train_time_str]
            time_multicvae = df_data[df_data.Model == 'MultiCVAE'][train_time_str]
            statistic, pvalue = f_oneway(list(time_carefl.values.flatten()),
                                         list(time_multicvae.values.flatten()),
                                         list(time_vcause.values.flatten()))
            print(f'p-value of the f_oneway for : {pvalue:.4f}')

            statistic, pvalue = ttest_ind(list(time_carefl.values.flatten()), list(time_vcause.values.flatten()))
            print(f'p-value of the T-test for VACA and CAREFL: {pvalue:.4f}')

            df_data = self.order_by_model(df_data)
            g = sns.catplot(x="Model", y=train_time_str, data=df_data, showfliers=False,
                            kind="box", legend=False,
                            )
            g.set_xlabels('')

            g.fig.suptitle(f'{d_name}')
            plt.show()
            g.savefig(os.path.join(self.save_dir, f'time_complexity_all_{d_name}.png'))

        # Number of parameters
        for d_name, df_data in df_time.groupby(['Dataset']):
            print(f'\nDataset: {d_name}')

            df_data = self.order_by_model(df_data)
            g = sns.catplot(x="Model", y=num_params_str, data=df_data, showfliers=False,
                            kind="box", legend=False,
                            )
            g.set_xlabels('')

            g.fig.suptitle(f'{d_name}')
            plt.show()
            g.savefig(os.path.join(self.save_dir, f'num_params_per_model_{d_name}.png'))

        print('\nPlotting training time versus number of parameters of the three models')

        ax = sns.scatterplot(data=df_time, x=num_params_str, y=train_time_str, hue="Model")

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

        ax.get_figure().savefig(os.path.join(self.save_dir, 'time_complexity_num_params.png'))

        # Compare time mean and std across datatasets and model

        print(df_time.groupby(['Dataset', 'Model']).agg(['mean', 'std'])[train_time_str])

    def time_complexity_VACA(self):

        df = self.df.copy()
        train_time_str = 'Total training time (min)'
        groupby_cols = ['Model', 'Dataset']
        metrics_cols = [train_time_str, 'model_params_num_hidden_dec']
        cols = [*groupby_cols, *metrics_cols]
        df = df[df['Model'] == 'VACA']

        df = df.rename(columns={"train_time_total": train_time_str})[cols]

        print(df.groupby(groupby_cols).agg(['mean', 'std', 'median'])[train_time_str])

    def VACA_decoder_layers(self, sem,
                            filter_=None):

        df = self.df.copy()
        df = df[df['Model'] == 'VACA']
        df = df[df['SEM'] == sem]
        if filter_ is not None:
            for key, values in filter_.items():
                df = df[df[key].isin(values)]

        df.rename(columns={"model_params_num_hidden_dec": r'$N_h$'}, inplace=True)
        groupby_cols = ['Dataset', r'$N_h$']
        metrics_cols = ['MMD Obs.', 'MMD Inter.']
        cols = [*groupby_cols, *metrics_cols]
        df = self.order_by_dataset(df)
        df = df[cols]

        print(df.groupby(groupby_cols).agg(['mean', 'std', 'count'])[metrics_cols] * 100)

        for nlayers in [0, 1, 2]:
            print(f'nlayers: {nlayers}')
            df_n = df[df[r'$N_h$'] == nlayers]
            my_str = ''
            data_str = ''
            for d_name in self.dataset_name_order:
                df_data = df_n[df_n.Dataset == d_name]
                if len(df_data) == 0: continue
                for met in metrics_cols:
                    mean = df_data[met].mean() * 100
                    std = df_data[met].std() * 100
                    my_str += ' & ' + f"{mean:.2f}" + ' $\pm$ ' + f"{std:.2f}"
                data_str += f" {d_name}"

            print(f'nlayers: {nlayers} dataset: {data_str}')
            print(f"{my_str} \\\\")

    def order_by_dataset(self, df):

        return self._order_df(df,
                              col_name='Dataset',
                              col_values_list=self.dataset_name_order)

    def order_by_model(self, df):

        return self._order_df(df,
                              col_name='Model',
                              col_values_list=self.model_name_order)

    def _order_df(self, df, col_name, col_values_list):
        df_out = df.copy()
        col_dict = {name: i for i, name in enumerate(col_values_list)}

        df_out[col_name] = df_out[col_name].replace(col_dict)
        df_out = df_out.sort_values(by=[col_name])
        col_dict = {i: name for i, name in enumerate(col_values_list)}
        df_out[col_name] = df_out[col_name].replace(col_dict)
        return df_out

    def VACA_dimension_z(self, limit_dim_z=None,
                         filter_=None):

        df_z = self.df[self.df.Model == 'VACA'].copy()

        if filter_ is not None:
            for key, value in filter_.items():
                df_z = df_z[df_z[key] == value]

        df_z.rename(columns={"model_params_z_dim": "dim(z)"}, inplace=True)
        df_z.rename(columns={"num_parameters": "Num. parameters"}, inplace=True)
        df_z = self.order_by_dataset(df_z)
        for dim_z, df_dim in df_z.groupby('dim(z)'):
            print(f'dim_z: {dim_z}')
            my_str = ''
            data_str = ''
            for d_name in self.dataset_name_order:
                df_data = df_dim[df_dim.Dataset == d_name]
                if len(df_data) == 0: continue
                data_str += f" {d_name}"

                for s_name in self.sem_name_order:
                    df_sem = df_data[df_data.SEM == s_name]
                    if len(df_sem) == 0: continue
                    data_str += f" {s_name}"
                    my_str += ' & ' + f"{df_sem['Num. parameters'].mean():.0f}"

            print(f'dim_z: {dim_z} dataset: {data_str}')
            print(f"{my_str} \\\\")

        if limit_dim_z: df_z = df_z[df_z['dim(z)'] <= limit_dim_z]

        print(f"Number of experiments: {len(df_z)}")
        metrics = ['MMD Obs.', 'MMD Inter.', 'MSE CF']
        df_g = df_z.groupby(['Dataset', 'SEM', 'dim(z)']).agg(['mean', 'std', 'count'])[metrics]

        print(df_g)

        return df_g

    def VACA_dimension_z_sem(self, limit_dim_z=None,
                             sem='non-linear',
                             filter_=None,
                             y_lim=None,
                             font_scale=1):

        cols_metrics = list(self.metrics_dict.keys())
        groupby_z = ['model_params_z_dim', 'Dataset', 'SEM']
        metrics_z = cols_metrics
        cols_z = [*groupby_z, *metrics_z]
        df_z = self.df[self.df.Model == 'VACA'].copy()
        df_z = df_z[df_z.SEM == sem]

        if filter_ is not None:
            for key, value in filter_.items():
                df_z = df_z[df_z[key] == value]

        df_z.rename(columns={"model_params_z_dim": "dim(z)"}, inplace=True)
        if limit_dim_z: df_z = df_z[df_z['dim(z)'] <= limit_dim_z]

        df_z = self.order_by_dataset(df_z)
        df_z.rename(columns={"num_parameters": "Num. parameters"}, inplace=True)

        print(f"Number of experiments: {len(df_z)}")
        metrics = ['MMD Obs.', 'MMD Inter.', 'MSE CF']
        # df_g = df_z.groupby(['dim(z)']).agg(['mean', 'std'])[metrics]

        print(df_z.groupby(['dim(z)']).agg(['mean', 'std', 'count'])[metrics])
        # x = 'dim(z)'
        # hue = 'Dataset'
        hue = 'dim(z)'
        x = 'Dataset'

        if font_scale != 1:
            sns.set(font_scale=font_scale)
            sns.set_style("white")
        for i, met in enumerate(metrics):
            g = sns.catplot(x=x, y=met, data=df_z, showfliers=False,
                            kind="box", legend=False, hue=hue
                            )
            # plt.legend(loc='best')

            if isinstance(y_lim, list):
                g.set(ylim=y_lim[i])
            g.fig.subplots_adjust(top=0.9)  # adjust the Figure in rp
            g.fig.suptitle(f'SEM: {sem}')
            plt.show()

            my_str = ''.join(filter(str.isalnum, met)).lower()

            g.savefig(os.path.join(self.save_dir, f'dimension_z_{my_str}_{sem}.png'))

        # Plot number of parameters
        fig, ax = plt.subplots()
        _ = sns.lineplot(x='dim(z)',
                         y='Num. parameters',
                         data=df_z,
                         legend=True,
                         hue='Dataset',
                         ax=ax)

        fig.subplots_adjust(top=0.9)  # adjust the Figure in rp

        fig.suptitle(f'SEM: {sem}')
        plt.show()

        my_str = ''.join(filter(str.isalnum, met)).lower()

        fig.savefig(os.path.join(self.save_dir, f'dimension_z_{sem}_num_params.png'))

        # print(df_z.groupby(['Dataset', 'dim(z)']).mean()[['Num. parameters']])

        return

    def cross_validate_nn(self, only_valid=True, model_name='VACA', metrics_to_use=[1, 2, 3, 4, 7, 8], debug=True):
        print('\nCross validating nn')
        cols_metrics = list(self.metrics_dict.keys())
        metrics_list = []
        for i, c in enumerate(cols_metrics):
            if i in metrics_to_use:
                metrics_list.append(c)
                add = 'True'
            else:
                add = 'False'
            print(f"({i}) [{add}] {c}")

        df = self.df.copy()
        if only_valid:
            df = self.filter_valid_configurations(df).copy()

        groupby = ['Dataset', 'SEM']
        if model_name == 'VACA':
            df = df[df.Model == 'VACA']
            df['model_params_h_dim_list_enc'] = df['model_params_h_dim_list_enc'].apply(lambda x: x.split('_')[0])
            df['model_params_h_dim_list_dec'] = df['model_params_h_dim_list_dec'].apply(lambda x: x.split('_')[0])
            df = df[df.model_params_h_dim_list_dec == df.model_params_h_dim_list_enc]
            df['model_params_h_dim_list'] = df['model_params_h_dim_list'].apply(lambda x: x.split('_')[-1])

            groupby.append('model_params_h_dim_list')
        elif model_name == 'CAREFL':
            df = df[df.Model == 'CAREFL']
            groupby.append('model_params_n_hidden')
        all_cols = [*groupby, *metrics_list]
        if debug:
            return all_cols, df

        df = df[all_cols]
        df[metrics_list] = df[metrics_list] * 100
        df_mean = df.groupby(groupby).mean()[metrics_list].applymap(lambda x: '{0:.2f}'.format(x))
        print(df_mean)
        df_std = df.groupby(groupby).std()[metrics_list].applymap(lambda x: '{0:.2f}'.format(x))
        print(df_std)

        df_comparison = df_mean + '$\pm$' + df_std
        table_file = os.path.join(self.save_dir, f'my_table_nn_{model_name}.tex')
        with open(table_file, 'w') as tf:
            tf.write(df_comparison.to_latex(escape=False))

        df_count = df.groupby(groupby).count()[metrics_list]
        print(df_count)

        table_file = os.path.join(self.save_dir, f'my_table_nn_count_{model_name}.tex')
        with open(table_file, 'w') as tf:
            tf.write(df_count.to_latex(escape=False))

        return


def print_cols(my_cols):
    for c in my_cols:
        print(c)
    print('')


def create_df_results(root_dir, add_time=False):
    experiment_results = []

    for json_file_name in glob.glob(os.path.join(root_dir, '**', 'output.json'), recursive=True):
        with open(json_file_name) as json_file:
            json_exper = json.load(json_file)
            json_exper['json_filename'] = json_file_name

        if add_time:

            json_file_name_time = os.path.join(os.path.dirname(json_file_name), 'time.json')
            if os.path.exists(json_file_name_time):
                with open(json_file_name_time) as json_file:
                    json_exper_time = json.load(json_file)
                    json_exper['train_time_total'] = json_exper_time['train_time_total']
                    json_exper['train_time_avg_per_epoch'] = json_exper_time['train_time_avg_per_epoch']
                    json_exper['train_epochs'] = json_exper['train_time_total'] / json_exper['train_time_avg_per_epoch']

        experiment_results.append(json_exper)

    return pd.DataFrame.from_dict(experiment_results)


def create_legend(label_list, color_list):
    return


def process_list_params(list_params):
    if isinstance(list_params, list) and len(list_params) > 0:
        return '_'.join([str(i) for i in list_params])
    else:
        return '0'


def get_number_hidden_layers(list_params):
    if isinstance(list_params, list):
        return len(list_params)
    else:
        return -1  # does not apply


def get_elements(my_list, my_and_filter, my_not_filter=[]):
    output = []

    for e in my_list:
        add = True
        for AND_F in my_and_filter:
            if AND_F not in e:
                add = False
                break

        for NOT_F in my_not_filter:
            if NOT_F in e:
                add = False
                break

        if add: output.append(e)

    return output


def get_unique_parameteres(columns_list, df_i, type_list=['model']):
    for c in get_elements(columns_list, type_list):
        if len(df_i[c].unique()) == 1: continue
        print(f"{c}")
        for i, u in enumerate(df_i[c].unique()):
            print(f"\t[{i}] {u}")
