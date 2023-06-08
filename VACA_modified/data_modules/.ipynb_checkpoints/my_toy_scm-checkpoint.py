import os

import pytorch_lightning as pl
import torch
from sklearn import preprocessing
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from torchvision import transforms as transform_lib

from data_modules._scalers import MaskedTensorLikelihoodScaler
from data_modules._scalers import MaskedTensorStandardScaler
from datasets.transforms import ToTensor
from utils.constants import Cte



from datasets.toy import create_toy_dataset
from utils.distributions import *


class MyToySCMDataModule(pl.LightningDataModule):
    name = 'het_scm'

    def __init__(
            self,
            data_dir: str = "./",
            dataset_name: str = Cte.CHAIN,
            equations_type: str = 'linear',
            nodes_to_intervene: list = None,
            structural_eq: dict = None,
            noises_distr: dict = None,
            adj_edges: dict = None,
            likelihood_names: str = None,
            lambda_: float = 0.05,
            num_samples_tr: int = 10000,
            num_workers: int = 0,
            normalize: str = None,
            normalize_A: str = None,
            seed: int = 42,
            batch_size: int = 32,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir

        self.equations_type = equations_type

        self.num_workers = num_workers
        self.normalize = normalize
        self.normalize_A = normalize_A
        self.scaler = None
        self.seed = seed
        self.batch_size = batch_size
        self.dataset_name = dataset_name

        self._shuffle_train = True




        root_dir = os.path.join(data_dir, 'VACA')

        datasets = []
        for split in ['train', 'valid', 'test']:
            if split == 'train':
                num_samples = num_samples_tr
            else:
                num_samples = int(0.5*num_samples_tr)

            dataset = create_toy_dataset(root_dir=root_dir,
                                         name=self.dataset_name,
                                         eq_type=equations_type,
                                         nodes_to_intervene=nodes_to_intervene,
                                         structural_eq=structural_eq,
                                         noises_distr=noises_distr,
                                         adj_edges=adj_edges,
                                         split=split,
                                         num_samples=num_samples,
                                         likelihood_names=likelihood_names,
                                         lambda_=lambda_)
            datasets.append(dataset)



        self.train_dataset = datasets[0]
        self.valid_dataset = datasets[1]
        self.test_dataset = datasets[2]

    @property
    def likelihood_list(self):
        return self.train_dataset.likelihood_list

    @property
    def topological_nodes(self):
        topological_nodes, _ = self.train_dataset.get_topological_nodes_pa()
        return topological_nodes

    @property
    def topological_parents(self):
        _, topological_pa = self.train_dataset.get_topological_nodes_pa()
        return topological_pa

    @property
    def node_dim(self):
        return self.train_dataset.node_dim

    @property
    def num_nodes(self):
        return self.train_dataset.num_nodes

    @property
    def edge_dimension(self):
        return self.train_dataset.num_edges

    @property
    def is_heterogeneous(self):
        return self.train_dataset.is_heterogeneous

    def set_shuffle_train(self, value):
        self._shuffle_train = value

    def get_node_dim_image(self):
        keys = self.train_dataset.nodes_list
        is_image = self.train_dataset.node_is_image()
        node_dims = self.train_dataset.get_node_dimensions()
        node_dim_image = {}
        for i, key in enumerate(keys):
            node_dim_image[key] = (node_dims[i], is_image[i])

        return node_dim_image

    def get_random_train_sampler(self):
        self.train_dataset.set_transform(self._default_transforms())

        def tmp_fn(num_samples):
            dataloader = DataLoader(self.train_dataset, batch_size=num_samples, shuffle=True)
            return next(iter(dataloader))

        return tmp_fn

    def get_deg(self, indegree=True, bincount=False):
        d_list = []
        idx = 1 if indegree else 0
        for data in self.train_dataset:
            d = degree(data.edge_index[idx], num_nodes=data.num_nodes, dtype=torch.long)
            d_list.append(d)

        d = torch.cat(d_list)
        if bincount:
            deg = torch.bincount(d, minlength=d.numel())
        else:
            deg = d

        return deg.float()

    def prepare_data(self, new_data=False):

        self.train_dataset.prepare_data(normalize_A=self.normalize_A, add_self_loop=True,
                                        new_data = new_data)
        self.valid_dataset.prepare_data(normalize_A=self.normalize_A, add_self_loop=True,
                                        new_data = new_data)
        self.test_dataset.prepare_data(normalize_A=self.normalize_A, add_self_loop=True,
                                       new_data = new_data)
        if self.normalize == 'std':
            self.scaler = MaskedTensorStandardScaler(list_dim_to_scale_x0=self.train_dataset.get_dim_to_scale_x0(),
                                                     list_dim_to_scale=self.train_dataset.get_dim_to_scale(),
                                                     total_num_dimensions=self.train_dataset.num_dimensions)
            self.scaler.fit(self.train_dataset.X0)
        elif self.normalize == 'lik':
            self.scaler = MaskedTensorLikelihoodScaler(likelihoods=self.train_dataset.likelihoods,
                                                       mask_x0=self.train_dataset.mask_X0[0, :])
            self.scaler.fit(self.train_dataset.X0)
        else:
            self.scaler = preprocessing.FunctionTransformer(func=lambda x: x,
                                                            inverse_func=lambda x: x)

    def train_dataloader(self):
        self.train_dataset.set_transform(self._default_transforms())
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self._shuffle_train,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        self.valid_dataset.set_transform(self._default_transforms())

        loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        self.test_dataset.set_transform(self._default_transforms())

        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        return loader

    def _default_transforms(self):
        if self.scaler is not None:
            return transform_lib.Compose(
                [lambda x: self.scaler.transform(x.reshape(1, self.train_dataset.total_num_dim_x0)), ToTensor()]
            )
        else:
            return ToTensor()

    def get_attributes_dict(self):
        return self.train_dataset.get_attributes_dict()

    def get_normalized_X(self, mode='test'):
        if mode == 'train':
            return self.scaler.transform(self.train_dataset.X.copy())
        elif mode == 'test':
            return self.scaler.transform(self.test_dataset.X.copy())
        elif mode == 'valid':
            return self.scaler.transform(self.valid_dataset.X.copy())
        else:
            raise NotImplementedError
