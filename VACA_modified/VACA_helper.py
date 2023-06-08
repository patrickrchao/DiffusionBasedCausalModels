import os
import utils.args_parser  as argtools
import pytorch_lightning as pl
import numpy as np
from utils.constants import Cte
from data_modules.my_toy_scm import MyToySCMDataModule
from utils.distributions import *
from data_modules.het_scm import HeterogeneousSCMDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from models._evaluator import MyEvaluator

from pathlib import Path


def create_data_from_data_params(dataset_params, new_data, n = None):
    if n is not None:
        dataset_params['num_samples_tr'] = n
    data_module = MyToySCMDataModule(**dataset_params)
    data_module.prepare_data(new_data = new_data)
    return data_module
    
def create_data(n,
                seed,
                structural_equations,
                noise_distributions,
                graph,
                name,
                equations_type,
                model_type,
                new_data = False):
    
    if model_type == "VACA":
        model_file = Path(__file__).parent / os.path.join('_params', 'model_vaca.yaml')
    elif model_type == "CAREFL":
        model_file = Path(__file__).parent / os.path.join('_params', 'model_carefl.yaml')
    else:
        raise Exception(f"Invalid model {model_type}")
    trainer_file = Path(__file__).parent / os.path.join('_params', 'trainer.yaml')


    cfg = argtools.parse_args(model_file)
    cfg.update(argtools.parse_args(trainer_file))
    # Config for new dataset
    cfg['dataset'] = {
        'params1': {},
        'params2': {}
    }

    cfg['dataset']['params1'] = {
        'data_dir': '../Data',
        'batch_size': 1000,
        'num_workers': 0
    }

    cfg['dataset']['params2'] = {
        'num_samples_tr': n,
        'equations_type': equations_type,
        'normalize': 'lik',
        'likelihood_names': 'd',
        'lambda_': 0.05,
        'normalize_A': None,
        
    }

    cfg['root_dir'] = 'results'
    cfg['seed'] = seed
    pl.seed_everything(cfg['seed'])

    cfg['dataset']['params'] = cfg['dataset']['params1'].copy()
    cfg['dataset']['params'].update(cfg['dataset']['params2'])

    cfg['dataset']['params']['data_dir'] = ''

    cfg['trainer']['limit_train_batches'] = 1.0
    cfg['trainer']['limit_val_batches'] = 1.0
    cfg['trainer']['limit_test_batches'] = 1.0
    cfg['trainer']['check_val_every_n_epoch'] = 10
    cfg['dataset']['params']['equations_type'] = equations_type 
    cfg['dataset']['name'] = name
    cfg['dataset']['params']['num_samples_tr'] = n
        
    intervene_nodes = []
    adj_edges = {}
    vaca_noise_dist = {}
    
    for node in graph.nodes():
        if graph.out_degree[node] > 0:
            intervene_nodes.append(node)
        adj_edges[node] = (list(graph.neighbors(node)))
        if type(noise_distributions[node]).__name__  == 'EmpiricalDistribution':
            data = noise_distributions[node].data
            vaca_noise_dist[node] = Empirical(data)
        else:
            
            noise_param = noise_distributions[node].parameters
            # Only works for normal distribution
        
            #assert noise_type == "norm", "Noise Distribution conversion to VACA only valid for Normal distributions"
            vaca_noise_dist[node] = Normal(mean = noise_param['loc'], 
                                       var = noise_param['scale']**2)
    
    dataset_params = cfg['dataset']['params'].copy()
    dataset_params['dataset_name'] = cfg['dataset']['name']

    dataset_params['nodes_to_intervene'] = intervene_nodes
    dataset_params['structural_eq'] = structural_equations
    dataset_params['noises_distr'] = vaca_noise_dist 
    dataset_params['adj_edges'] = adj_edges
    dataset_params['nodes_list'] = sorted(graph.nodes())
    data_module = create_data_from_data_params(dataset_params, new_data)
    return data_module, cfg, dataset_params

def create_model(cfg, data_module):
 
    model = None
    model_params = cfg['model']['params'].copy()

    #print(f"\nUsing model: {cfg['model']['name']}")


    # VACA
    if cfg['model']['name'] == Cte.VACA:
        from models.vaca.vaca import VACA

        model_params['is_heterogeneous'] = data_module.is_heterogeneous
        model_params['likelihood_x'] = data_module.likelihood_list

        model_params['deg'] = data_module.get_deg(indegree=True)
        model_params['num_nodes'] = data_module.num_nodes
        model_params['edge_dim'] = data_module.edge_dimension
        model_params['scaler'] = data_module.scaler

        model = VACA(**model_params)
        model.set_random_train_sampler(data_module.get_random_train_sampler())
    # VACA with PIWAE
    elif cfg['model']['name'] == Cte.VACA_PIWAE:
        from models.vaca.vaca_piwae import VACA_PIWAE

        model_params['is_heterogeneous'] = data_module.is_heterogeneous

        model_params['likelihood_x'] = data_module.likelihood_list

        model_params['deg'] = data_module.get_deg(indegree=True)
        model_params['num_nodes'] = data_module.num_nodes
        model_params['edge_dim'] = data_module.edge_dimension
        model_params['scaler'] = data_module.scaler

        model = VACA_PIWAE(**model_params)
        model.set_random_train_sampler(data_module.get_random_train_sampler())


    # MultiCVAE
    elif cfg['model']['name'] == Cte.MCVAE:
        from models.multicvae.multicvae import MCVAE

        model_params['likelihood_x'] = data_module.likelihood_list

        model_params['topological_node_dims'] = data_module.train_dataset.get_node_columns_in_X()
        model_params['topological_parents'] = data_module.topological_parents
        model_params['scaler'] = data_module.scaler
        model_params['num_epochs_per_nodes'] = int(
            np.floor((cfg['trainer']['max_epochs'] / len(data_module.topological_nodes))))
        model = MCVAE(**model_params)
        model.set_random_train_sampler(data_module.get_random_train_sampler())
        cfg['early_stopping'] = False

    # CAREFL
    elif cfg['model']['name'] == Cte.CAREFL:
        from models.carefl.carefl import CAREFL

        model_params['node_per_dimension_list'] = data_module.train_dataset.node_per_dimension_list
        model_params['scaler'] = data_module.scaler
        model = CAREFL(**model_params)

    model.set_optim_params(optim_params=cfg['optimizer'],
                           sched_params=cfg['scheduler'])
    return model

def fit_model(model, cfg, data_module):
    

    is_training = True
    load = False

   # print(f'Is training activated? {is_training}')
   # print(f'Is loading activated? {load}')
    

    save_dir = argtools.mkdir(os.path.join(cfg['root_dir'],
                                           argtools.get_experiment_folder(cfg),
                                           str(cfg['seed']),str(cfg['dataset']['params']['num_samples_tr']),
                                           str(cfg['trainer']['max_epochs'])))

    logger = TensorBoardLogger(save_dir=save_dir, name='logs', default_hp_metric=False)

    out = logger.log_hyperparams(argtools.flatten_cfg(cfg))

    save_dir_ckpt = argtools.mkdir(os.path.join(save_dir, 'ckpt'))
    if load:
        try:
            ckpt_file = argtools.newest(save_dir_ckpt)
            if ckpt_file is not None:
                is_training = False
        except:
            ckpt_file = None
    else:
        ckpt_file = None
    callbacks = []



    evaluator = MyEvaluator(model=model,
                            intervention_list=data_module.train_dataset.get_intervention_list(),
                            scaler=data_module.scaler
                            )
    model.set_my_evaluator(evaluator=evaluator)

    if is_training:
        checkpoint = ModelCheckpoint(period=1,
                                     monitor=model.monitor(),
                                     mode=model.monitor_mode(),
                                     save_top_k=1,
                                     save_last=True,
                                     filename='checkpoint-{epoch:02d}',
                                     dirpath=save_dir_ckpt)
        callbacks = [checkpoint]


        if cfg['early_stopping']:
            early_stopping = EarlyStopping(model.monitor(), mode=model.monitor_mode(), min_delta=0.0, patience=50)
            callbacks.append(early_stopping)
        trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg['trainer'])

    if load:
        if ckpt_file is not None:
           
            if is_training:
                trainer = pl.Trainer(logger=logger, callbacks=callbacks, resume_from_checkpoint=ckpt_file,
                                 **cfg['trainer'])
            else:

                model = model.load_from_checkpoint(ckpt_file, **model_params)
                evaluator.set_model(model)
                model.set_my_evaluator(evaluator=evaluator)

                if cfg['model']['name'] in [Cte.VACA_PIWAE, Cte.VACA, Cte.MCVAE]:
                    model.set_random_train_sampler(data_module.get_random_train_sampler())
    path = os.path.join(save_dir,"logs")

    if not os.path.exists(path):
        os.makedirs(path)

    if is_training:
        trainer.fit(model, data_module)
        # save_yaml(model.get_arguments(), file_path=os.path.join(save_dir, 'hparams_model.yaml'))
       # argtools.save_yaml(cfg, file_path=os.path.join(save_dir, 'hparams_full.yaml'))
        # %% Testing
    