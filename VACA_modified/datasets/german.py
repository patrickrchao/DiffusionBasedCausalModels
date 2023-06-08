import os

from datasets._heterogeneous import HeterogeneousSCM
from utils.distributions import *


class GermanSCM(HeterogeneousSCM):

    def __init__(self, root_dir,
                 split: str = 'train',
                 num_samples_tr: int = 800,
                 lambda_: float = 0.05,
                 transform=None,
                 ):
        assert split in ['train', 'valid', 'test']

        self.name = 'german'
        self.split = split
        self.num_samples_tr = num_samples_tr
        self.X_train = np.load(os.path.join(root_dir, 'german_data', 'train_1_X.npy'))
        self.X_valid = np.load(os.path.join(root_dir, 'german_data', 'test_valid_1_X.npy'))

        self.Y = None
        super().__init__(root_dir=root_dir,
                         transform=transform,
                         nodes_to_intervene=['age'],
                         structural_eq=None,
                         noises_distr=None,
                         nodes_list=['sex', 'age', 'R', 'S'],
                         adj_edges={'sex': ['R', 'S'],  # A#excluding 16
                                    'age': ['R', 'S'],  # Age
                                    'R': [],
                                    'S': [],
                                    },
                         lambda_=lambda_,
                         )

    @property
    def likelihoods(self):
        likelihoods_tmp = []
        # lik_node_sex = [self._get_lik('b',dim=1,normalize='dim')]
        # likelihoods_tmp.append(lik_node_sex)
        lik_node_sex = [self._get_lik('d', dim=1, normalize=None)]
        likelihoods_tmp.append(lik_node_sex)

        lik_node_age = [self._get_lik('d', dim=1, normalize='dim')]
        likelihoods_tmp.append(lik_node_age)

        lik_node_R = [self._get_lik('d', dim=2, normalize='dim')]
        likelihoods_tmp.append(lik_node_R)

        lik_node_S = [self._get_lik('c', dim=3, normalize='dim'),
                      self._get_lik('c', dim=5, normalize='dim'),
                      self._get_lik('c', dim=4, normalize='dim')]
        likelihoods_tmp.append(lik_node_S)

        return likelihoods_tmp

    def _create_data(self):
        X = np.concatenate([self.X_train, self.X_valid], axis=0)
        if self.split == 'train':
            self.X = X[:self.num_samples_tr]
        elif self.split == 'valid':
            num_samples = (X.shape[0] - self.num_samples_tr) // 2
            self.X = X[self.num_samples_tr:(self.num_samples_tr + num_samples)]
        elif self.split == 'test':
            num_samples = (X.shape[0] - self.num_samples_tr) // 2
            self.X = X[-num_samples:]
        self.Y = self.X[:, -1].copy()
        self.X = self.X[:, :-1]
        self.U = np.zeros([self.X.shape[0], 1])

    def node_is_image(self):
        return [False, False, False, False]

    def is_toy(self):
        return False

    def get_attributes_dict(self):
        # TODO:  Makes this more general
        unfair_attributes = list(range(2, 16))  # R, S
        fair_attributes = [1]  # Age
        sensitive_attributes = [0]  # Sex
        return {'unfair_attributes': unfair_attributes,
                'fair_attributes': fair_attributes,
                'sens_attributes': sensitive_attributes}
