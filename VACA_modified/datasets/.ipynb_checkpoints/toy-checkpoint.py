from datasets._heterogeneous import HeterogeneousSCM

flatten = lambda t: [item for sublist in t for item in sublist]

from utils.distributions import *
from utils.constants import Cte
import os
from utils.args_parser import mkdir


class ToySCM(HeterogeneousSCM):

    def __init__(self, root_dir,
                 name: str = 'chain',
                 eq_type: str = Cte.LINEAR,
                 nodes_to_intervene: list = None,
                 structural_eq: dict = None,
                 noises_distr: dict = None,
                 adj_edges: dict = None,
                 split: str = 'train',
                 num_samples: int = 5000,
                 likelihood_names: str = 'd_d_cb',
                 lambda_: float = 0.05,
                 transform=None,
                 nodes_list=None,
                 ):
        assert split in ['train', 'valid', 'test']

        self.split = split
        self.name = name
        self._num_samples = num_samples
        self.eq_type = eq_type
        num_nodes = len(structural_eq)

        likelihood_names = likelihood_names.split('_')
        if len(likelihood_names) == 1:
            likelihood_names = [likelihood_names[0], ] * num_nodes
        self.likelihood_names = likelihood_names
        assert num_nodes == len(noises_distr)
        assert num_nodes == len(adj_edges)
        assert num_nodes == len(self.likelihood_names)

        if nodes_list is None:
            nodes_list = [f'x{i + 1}' for i in range(num_nodes)]
            
        assert set(structural_eq.keys()) ==  set(noises_distr.keys()), 'Keys for the SE and Noise distribution should be the same'
        #for key_eq, key_n in zip(structural_eq.keys(), noises_distr.keys()):
        #    assert key_eq == key_n, 'Keys for the SE and Noise distribution should be the same'
        super().__init__(root_dir=root_dir,
                         transform=transform,
                         nodes_to_intervene=nodes_to_intervene,
                         nodes_list=nodes_list,
                         adj_edges=adj_edges,
                         structural_eq=structural_eq,
                         noises_distr=noises_distr,
                         lambda_=lambda_
                         )

    @property
    def likelihoods(self):
        likelihoods_tmp = []

        for i, lik_name in enumerate(self.likelihood_names):
            likelihoods_tmp.append([self._get_lik(lik_name,
                                                  dim=1,
                                                  normalize='dim')])

        return likelihoods_tmp

    @property
    def std_list(self):
        return [-1, -0.5, 0, 0.5, 1]

    def _create_data(self, new_data):
        X = np.zeros([self._num_samples, self.num_dimensions])
        U = np.zeros([self._num_samples, self.num_nodes])

        folder = mkdir(os.path.join(self.root_dir, f'{self.name}_{self.eq_type}'))
        
        X_file = os.path.join(folder, f'{self.split}_{self._num_samples}_X.npy')
        U_file = os.path.join(folder, f'{self.split}_{self._num_samples}_U.npy')
        if (not new_data) and os.path.exists(X_file) and os.path.exists(U_file):
            X = np.load(X_file)
            U = np.load(U_file)
            np.save(X_file, X)
            np.save(U_file, U)
        else:
            for i in range(self._num_samples):
                x, u = self.sample()
                X[i, :] = x
                U[i, :] = u


        self.X = X.astype(np.float32)
        self.U = U.astype(np.float32)

    def node_is_image(self):
        return [False for _ in self.num_nodes]


def create_toy_dataset(root_dir,
                       name: str = 'chain',
                       eq_type: str = Cte.LINEAR,
                       nodes_to_intervene: list = None,
                       structural_eq: dict = None,
                       noises_distr: dict = None,
                       adj_edges: dict = None,
                       split: str = 'train',
                       num_samples: int = 5000,
                       likelihood_names: str = 'd_d_cb',
                       lambda_: float = 0.05,
                       transform=None):

    return ToySCM(root_dir,
                  name,
                  eq_type,
                  nodes_to_intervene,
                  structural_eq,
                  noises_distr,
                  adj_edges,
                  split,
                  num_samples,
                  likelihood_names,
                  lambda_=lambda_,
                  transform=transform)
