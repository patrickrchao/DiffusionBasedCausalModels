from datasets.toy import create_toy_dataset
from utils.distributions import *
dataset = create_toy_dataset(root_dir='./my_custom_datasets',
                             name='2graph',
                             eq_type='linear',
                             nodes_to_intervene=['x1'],
                             structural_eq={'x1': lambda u1: u1,
                                            'x2': lambda u2, x1: u2 + x1},
                             noises_distr={'x1': Normal(0,1),
                                           'x2': Normal(0,1)},
                             adj_edges={'x1': ['x2'],
                                        'x2': []},
                             split='train',
                             num_samples=5000,
                             likelihood_names='d_d',
                             lambda_=0.05)
