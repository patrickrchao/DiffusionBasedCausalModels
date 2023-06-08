"""
This file creates all the Toy datasets included in this repository
"""

from data_modules.het_scm import HeterogeneousSCMDataModule
import os
from utils.constants import Cte

for data_name in [Cte.COLLIDER, Cte.MGRAPH, Cte.TRIANGLE, Cte.CHAIN]:
    for sem in [Cte.LINEAR, Cte.NONLINEAR, Cte.NONADDITIVE]:
        print(f"\n\n{data_name} {sem}")
        if data_name == Cte.MGRAPH:
            likelihood_names = 'd_d_d_d_d'
        else:
            likelihood_names = 'd_d_d'

        dm = HeterogeneousSCMDataModule(data_dir=os.path.join('..', 'Data'),
                                        dataset_name=data_name,
                                        equations_type=sem,
                                        num_samples_tr=5000,
                                        num_workers=0,
                                        normalize='lik',
                                        likelihood_names=likelihood_names,
                                        lambda_=0.05,
                                        normalize_A=None,
                                        seed=42,
                                        batch_size=4)

        dm.prepare_data()

        dataset = dm.train_dataset

        print(f"X: {dataset.X.shape}")
        print(f"U: {dataset.U.shape}")



# %%

data_name = Cte.LOAN
sem = Cte.LINEAR
print(f"{data_name} {sem}")
dm = HeterogeneousSCMDataModule(data_dir=os.path.join('..', 'Data'),
                                dataset_name=data_name,
                                equations_type=sem,
                                num_samples_tr=5000,
                                num_workers=0,
                                normalize='lik',
                                likelihood_names='d_d_d_d_d_d_d',
                                lambda_=0.05,
                                normalize_A=None,
                                seed=42,
                                batch_size=4)

dm.prepare_data()

print(dm.train_dataset.name)

dataset = dm.train_dataset

print(f"X: {dataset.X.shape}")
print(f"U: {dataset.U.shape}")


# %%
data_name = Cte.ADULT
sem = Cte.LINEAR
print(f"{data_name} {sem}")
dm = HeterogeneousSCMDataModule(data_dir=os.path.join('..', 'Data'),
                                dataset_name=data_name,
                                equations_type=sem,
                                num_samples_tr=5000,
                                num_workers=0,
                                normalize='lik',
                                likelihood_names='d_d_d_d_d_d_d',
                                lambda_=0.05,
                                normalize_A=None,
                                seed=42,
                                batch_size=4)

dm.prepare_data()

print(dm.train_dataset.name)

dataset = dm.train_dataset

print(f"X: {dataset.X.shape}")
print(f"U: {dataset.U.shape}")