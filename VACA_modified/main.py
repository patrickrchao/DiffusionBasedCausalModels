import argparse
import json
import os
import warnings

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

import utils.args_parser  as argtools
import utils.tools as utools
from utils.constants import Cte

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset_file', default='_params/dataset_toy.yaml', type=str,
                    help='path to configuration file for the dataset')
parser.add_argument('--model_file', default='_params/model_mcvae.yaml', type=str,
                    help='path to configuration file for the dataset')
parser.add_argument('--trainer_file', default='_params/trainer.yaml', type=str,
                    help='path to configuration file for the training')
parser.add_argument('--yaml_file', default='', type=str, help='path to trained model configuration')
parser.add_argument('-d', '--dataset_dict', action=argtools.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                    help='manually define dataset configurations as string: KEY1=VALUE1+KEY2=VALUE2+...')
parser.add_argument('-m', '--model_dict', action=argtools.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                    help='manually define model configurations as string: KEY1=VALUE1+KEY2=VALUE2+...')
parser.add_argument('-o', '--optim_dict', action=argtools.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                    help='manually define optimizer configurations as string: KEY1=VALUE1+KEY2=VALUE2+...')
parser.add_argument('-t', '--trainer_dict', action=argtools.StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...",
                    help='manually define trainer configurations as string: KEY1=VALUE1+KEY2=VALUE2+...')
parser.add_argument('-s', '--seed', default=-1, type=int, help='set random seed, default: random')
parser.add_argument('-r', '--root_dir', default='', type=str, help='directory for storing results')
parser.add_argument('--data_dir', default='', type=str, help='data directory')
parser.add_argument('-i', '--is_training', default=1, type=int, help='run with training (1) or without training (0)')
parser.add_argument('-f', '--eval_fair', default=False, action="store_true",
                    help='run code with counterfactual fairness experiment (only for German dataset), default: False')
parser.add_argument('--show_results', default=True, action="store_true",
                    help='run with evaluation (1) or without(0), default: 1')

parser.add_argument('--plots', default=0, type=int, help='run code with plotting (1) or without (0), default: 0')

args = parser.parse_args()

# %%
if args.yaml_file == '':
    cfg = argtools.parse_args(args.dataset_file)
    cfg.update(argtools.parse_args(args.model_file))
    cfg.update(argtools.parse_args(args.trainer_file))
else:
    cfg = argtools.parse_args(args.yaml_file)
if len(args.root_dir) > 0:  cfg['root_dir'] = args.root_dir
if int(args.seed) >= 0:
    cfg['seed'] = int(args.seed)

# %%
pl.seed_everything(cfg['seed'])
if args.dataset_dict is not None: cfg['dataset']['params2'].update(args.dataset_dict)
if args.model_dict is not None: cfg['model']['params'].update(args.model_dict)
if args.optim_dict is not None: cfg['optimizer']['params'].update(args.optim_dict)
if args.trainer_dict is not None: cfg['trainer'].update(args.trainer_dict)

if isinstance(cfg['trainer']['gpus'], int):
    cfg['trainer']['auto_select_gpus'] = False
    cfg['trainer']['gpus'] = -1

cfg['dataset']['params'] = cfg['dataset']['params1'].copy()
cfg['dataset']['params'].update(cfg['dataset']['params2'])

if len(args.data_dir) > 0:
    cfg['dataset']['params']['data_dir'] = args.data_dir

print(args.dataset_dict)
print(cfg['dataset']['params'])
print(cfg['model']['params'])

# %% Load dataset

data_module = None

if cfg['dataset']['name'] in Cte.DATASET_LIST:
    from data_modules.het_scm import HeterogeneousSCMDataModule

    dataset_params = cfg['dataset']['params'].copy()
    dataset_params['dataset_name'] = cfg['dataset']['name']

    data_module = HeterogeneousSCMDataModule(**dataset_params)
    data_module.prepare_data()

assert data_module is not None, cfg['dataset']

# %% Load model
model = None
model_params = cfg['model']['params'].copy()
# utools.blockPrint()

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
elif cfg['model']['name'] == Cte.CARELF:
    from models.carefl.carefl import CAREFL

    model_params['node_per_dimension_list'] = data_module.train_dataset.node_per_dimension_list
    model_params['scaler'] = data_module.scaler
    model = CAREFL(**model_params)
assert model is not None
utools.enablePrint()

model.summarize()
model.set_optim_params(optim_params=cfg['optimizer'],
                       sched_params=cfg['scheduler'])

# %% Evaluator

evaluator = None

if cfg['dataset']['name'] in Cte.DATASET_LIST:
    from models._evaluator import MyEvaluator

    evaluator = MyEvaluator(model=model,
                            intervention_list=data_module.train_dataset.get_intervention_list(),
                            scaler=data_module.scaler
                            )

assert evaluator is not None

model.set_my_evaluator(evaluator=evaluator)

# %% Prepare training
if args.yaml_file == '':
    if (cfg['dataset']['name'] in [Cte.GERMAN]) and (cfg['dataset']['params3']['train_kfold'] == True):
        save_dir = argtools.mkdir(os.path.join(cfg['root_dir'],
                                               argtools.get_experiment_folder(cfg),
                                               str(cfg['seed']), str(cfg['dataset']['params3']['kfold_idx'])))
    else:
        save_dir = argtools.mkdir(os.path.join(cfg['root_dir'],
                                               argtools.get_experiment_folder(cfg),
                                               str(cfg['seed'])))
else:
    save_dir = os.path.join(*args.yaml_file.split('/')[:-1])
print(f'Save dir: {save_dir}')
# trainer = pl.Trainer(**cfg['model'])
logger = TensorBoardLogger(save_dir=save_dir, name='logs', default_hp_metric=False)
out = logger.log_hyperparams(argtools.flatten_cfg(cfg))

save_dir_ckpt = argtools.mkdir(os.path.join(save_dir, 'ckpt'))
ckpt_file = argtools.newest(save_dir_ckpt)
callbacks = []
if args.is_training == 1:

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

    if ckpt_file is not None:
        print(f'Loading model training: {ckpt_file}')
        trainer = pl.Trainer(logger=logger, callbacks=callbacks, resume_from_checkpoint=ckpt_file,
                             **cfg['trainer'])
    else:

        trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg['trainer'])

    # %% Train

    trainer.fit(model, data_module)
    # save_yaml(model.get_arguments(), file_path=os.path.join(save_dir, 'hparams_model.yaml'))
    argtools.save_yaml(cfg, file_path=os.path.join(save_dir, 'hparams_full.yaml'))
    # %% Testing

else:
    # %% Testing
    trainer = pl.Trainer()
    print('\nLoading from: ')
    print(ckpt_file)

    model = model.load_from_checkpoint(ckpt_file, **model_params)
    evaluator.set_model(model)
    model.set_my_evaluator(evaluator=evaluator)

    if cfg['model']['name'] in [Cte.VACA_PIWAE, Cte.VACA, Cte.MCVAE]:
        model.set_random_train_sampler(data_module.get_random_train_sampler())

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = int(sum([np.prod(p.size()) for p in model_parameters]))

print(f'Model parameters: {params}')
model.eval()
model.freeze()  # IMPORTANT

if args.show_results:
    output_valid = model.evaluate(dataloader=data_module.val_dataloader(),
                                  name='valid',
                                  save_dir=save_dir,
                                  plots=False)
    output_test = model.evaluate(dataloader=data_module.test_dataloader(),
                                 name='test',
                                 save_dir=save_dir,
                                 plots=args.plots)
    output_valid.update(output_test)

    output_valid.update(argtools.flatten_cfg(cfg))
    output_valid.update({'ckpt_file': ckpt_file,
                         'num_parameters': params})

    with open(os.path.join(save_dir, 'output.json'), 'w') as f:
        json.dump(output_valid, f)
    print(f'Experiment folder: {save_dir}')

if args.eval_fair:
    assert cfg['dataset']['name'] in [Cte.GERMAN], "counterfactual fairness not implemented for dataset"

    output_fairness = model.my_cf_fairness(data_module=data_module,
                                           save_dir=save_dir)

    output_fairness.update(argtools.flatten_cfg(cfg))
    output_fairness.update({'ckpt_file': ckpt_file,
                            'num_parameters': params})

    with open(os.path.join(save_dir, 'fairness.json'), 'w') as f:
        json.dump(output_fairness, f)
    print(f'Experiment folder: {save_dir}')
