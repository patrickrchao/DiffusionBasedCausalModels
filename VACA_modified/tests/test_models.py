# %%
from data_modules.het_scm import HeterogeneousSCMDataModule

data_module = HeterogeneousSCMDataModule(data_dir="../Data",
                                         dataset_name='triangle',
                                         num_workers=0,
                                         num_samples_tr=5000,
                                         batch_size=256,
                                         likelihood_names='d_d_d',
                                         normalize='lik',
                                         lambda_=0.05,
                                         equations_type='linear')

data_module.prepare_data()
print(data_module.edge_dimension)

data_loader = data_module.train_dataloader()
scaler = data_module.scaler
batch = next(iter(data_loader))

# %%
from utils.constants import Cte
model_list =  []
from models.carefl.carefl import CAREFL

model = CAREFL(node_per_dimension_list=data_module.train_dataset.node_per_dimension_list,
               distr_z='laplace',
               flow_net_class='mlp',
               flow_architecture='spline',
               n_layers=1,
               n_hidden=1,
               parity=False,
               scaler=data_module.scaler,
               init=None)


model_list.append(model)

from models.multicvae.multicvae import MCVAE

model = MCVAE(h_dim_list_dec=[8,8],
                 h_dim_list_enc=[8],
                 z_dim=4,
                 drop_rate=0.0,
                 act_name=Cte.RELU,
                 likelihood_x=data_module.likelihood_list,
                 distr_z='normal',
                 num_epochs_per_nodes=50,
                 topological_node_dims=data_module.train_dataset.get_node_columns_in_X(),
                 topological_parents=data_module.topological_parents,
                 scaler=data_module.scaler)


model_list.append(model)

from models.vaca.vaca import VACA

model = VACA(h_dim_list_dec=[8,8],
                 h_dim_list_enc=[8],
                 z_dim=4,
                 m_layers = 1,
                 deg = data_module.get_deg(indegree=True),
                 edge_dim= data_module.edge_dimension,
                 num_nodes=data_module.num_nodes,
                 beta = 1.0,
                 annealing_beta = False,
                 residual = 0,  # Only PNA architecture
                 drop_rate = 0.0,
                 dropout_adj_rate = 0.0,
                 dropout_adj_pa_rate = 0.0,
                 dropout_adj_pa_prob_keep_self = 0.0,
                 keep_self_loops = True,
                 dropout_adj_T = 0,  # Epoch to start the dropout_adj_T
                 act_name = Cte.RELU,
                 likelihood_x = data_module.likelihood_list,
                 distr_z = 'normal',
                 architecture = 'pna',  # PNA, DGNN, DPNA
                 estimator = 'elbo',
                 K=1,  # Only for IWAE estimator
                 scaler = data_module.scaler,
                 init = None,
                 is_heterogeneous=data_module.is_heterogeneous)


model_list.append(model)




# %%

from models._evaluator import MyEvaluator

x_I={data_module.train_dataset.nodes_list[0]: 2}

for model in model_list[1:]:
    print(f"\n\nMODEL: {model._get_name()}\n\n")
    evaluator = MyEvaluator(model=model,
                            intervention_list=data_module.train_dataset.get_intervention_list(),
                            scaler=data_module.scaler
                            )

    model.set_my_evaluator(evaluator=evaluator)

    if model._get_name() != 'CAREFL':
        model.set_random_train_sampler(data_module.get_random_train_sampler())

    output = model.get_objective_metrics(data_loader=data_loader, name='test')


    z, x, x_real = model.get_observational_distr(data_loader)

    print(f"z: {z.shape}")
    print(f"x: {x.shape}")
    print(f"x_real: {x_real.shape}")



    z, x, x_real = model.get_reconstruction_distr(data_loader)


    x_gener_dict_out, x_real_dict_out = model.get_interventional_distr(data_loader,
                                                                       x_I=x_I,
                                                                       use_aggregated_posterior=False)

    x_cf_gener_dict, x_cf_real_dict, x_factual_dict = model.get_counterfactual_distr(data_loader,
                                                                                     x_I=x_I,
                                                                                     is_noise=False)



    output = model.evaluate(data_loader, name='test', save_dir='.')
