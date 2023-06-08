
import os
import numpy as np
from utils._errors import IsHeterogeneousError
def print_dataset_details(dataset):
    print('')
    print(f"DATASET {dataset.name}")
    dataset.prepare_data()
    print(f"\ndataset.num_nodes")
    print(dataset.num_nodes)
    print(f"\ndataset.node_dim")
    try:
        print(dataset.node_dim)
    except IsHeterogeneousError:
        print('Dataset is heterogeneous! You will get an error if you use dataset.node_dim')

    print(f"\ndataset.num_parameters")
    print(dataset.num_parameters)

    print(f"\ndataset.dim_of_x_in_x0")
    print(dataset.dim_of_x_in_x0)

    print(f"\ndataset._get_x0_dim_of_node_name({dataset.nodes_list[0]})")

    print(dataset._get_x0_dim_of_node_name(dataset.nodes_list[0]))


    node0_name = dataset.nodes_list[0]

    print(f"\nIntervention {node0_name}=0")
    x_I = {node0_name: 0}
    dataset.set_intervention(x_I=x_I)
    batch = dataset.__getitem__(0)
    x = batch.x.reshape(1, -1)
    print(f"data.x[0]={x[0]}")
    x_i = batch.x_i.reshape(1, -1)
    print(f"data.x_i[0]={x_i[0]}")

    x_sample_intervention = dataset.sample_intervention(x_I=x_I,
                                                        n_samples=2)

    if dataset.has_ground_truth:
        print(f"\nGenerating 2 interventional samples {node0_name}=0")
        with np.printoptions(precision=2, suppress=True):
            print(f"x_sample_intervention=\n{x_sample_intervention}")

        print(f"\nGetting 2 counterfactual samples {node0_name}=0")
        batch_1 = dataset.__getitem__(1)
        x = np.concatenate([batch.x.reshape(1, -1),
                            batch_1.x.reshape(1, -1)])
        u = np.concatenate([batch.u.reshape(1, -1),
                            batch_1.u.reshape(1, -1)])
        x_cf = dataset.get_counterfactual(x_factual=x,
                                          u_factual=u,
                                          x_I=x_I)
        with np.printoptions(precision=2, suppress=True):
            for i in range(2):
                print(f"x[{i}]={x[i]}")
                print(f"x_cf[{i}]={x_cf[i]}")


root_dir = os.path.join('.', '_data')

# %% Collider
from datasets.collider import ColliderSCM
from utils.constants import Cte

dataset = ColliderSCM(root_dir=root_dir,
                 split = 'train',
                 num_samples = 5000,
                 equations_type=Cte.LINEAR,
                 likelihood_names='d_d_d',
                 transform=None)

print_dataset_details(dataset)

# %% M-graph
from datasets.mgraph import MGraphSCM

dataset = MGraphSCM(root_dir=root_dir,
                 split = 'train',
                 num_samples = 5000,
                 equations_type=Cte.LINEAR,
                 likelihood_names='d_d_d_d_d',
                 transform=None)

print_dataset_details(dataset)


# %% Triangle
from datasets.triangle import TriangleSCM

dataset = TriangleSCM(root_dir=root_dir,
                 split = 'train',
                 num_samples = 5000,
                 equations_type=Cte.LINEAR,
                 likelihood_names='d_d_d',
                 transform=None)

print_dataset_details(dataset)

# %% Chain
from datasets.chain import ChainSCM
dataset = ChainSCM(root_dir=root_dir,
                 split = 'train',
                 num_samples = 5000,
                 equations_type=Cte.LINEAR,
                 likelihood_names='d_d_d',
                 transform=None)

print_dataset_details(dataset)

# %% Loan
from datasets.loan import LoanSCM
from utils.constants import Cte

dataset = LoanSCM(root_dir=root_dir,
                 split = 'train',
                 num_samples = 5000,
                 equations_type=Cte.LINEAR,
                 likelihood_names='b_d_d_d_d_d_d',
                 transform=None)

print_dataset_details(dataset)


# %% German
from datasets.german import GermanSCM
from utils.constants import Cte

dataset = GermanSCM(root_dir=root_dir,
                 split = 'train',
                 num_samples_tr = 800,
                 lambda_ = 0.05,
                 transform=None)

print_dataset_details(dataset)


# %% Adult

from datasets.adult import AdultSCM
from utils.constants import Cte
dataset = AdultSCM(root_dir=root_dir,
                 split = 'train',
                 num_samples = 5000,
                 equations_type=Cte.LINEAR,
                 likelihood_names = 'c_d_c_b_d_d_c_c_c_c_d',
                 transform=None)

print_dataset_details(dataset)
