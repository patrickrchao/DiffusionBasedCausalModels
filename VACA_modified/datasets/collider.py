from datasets.toy import ToySCM
from utils.constants import Cte
from utils.distributions import *


class ColliderSCM(ToySCM):

    def __init__(self, root_dir,
                 split: str = 'train',
                 num_samples: int = 5000,
                 equations_type=Cte.LINEAR,
                 likelihood_names: str = 'd_d_cb',
                 transform=None,
                 **kwargs
                 ):

        if equations_type == Cte.LINEAR:
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2: u2,
                'x3': lambda u3, x1, x2: 0.05 * x1 + 0.25 * x2 + u3,
            }

            noises_distr = {
                'x1': MixtureOfGaussians(probs=[0.5, 0.5], means=[-2, 1.5], vars=[1.5, 1]),
                'x2': Normal(0, 1),
                'x3': Normal(0, 1),
            }

        elif equations_type == Cte.NONLINEAR:
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2: u2,
                'x3': lambda u3, x1, x2: 0.05 * x1 + 0.25 * x2 ** 2 + u3,
            }

            noises_distr = {
                'x1': MixtureOfGaussians(probs=[0.5, 0.5], means=[-2, 1.5], vars=[1.5, 1]),
                'x2': Normal(0, 0.1),
                'x3': Normal(0, 1),
            }

        elif equations_type == Cte.NONADDITIVE:
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2: u2,
                'x3': lambda u3, x1, x2: -1 + 0.1 * np.sign(u3) * (x1 ** 2 + x2 ** 2) * u3,
            }

            noises_distr = {
                'x1': MixtureOfGaussians(probs=[0.5, 0.5], means=[-2.5, 2.5], vars=[1, 1]),
                'x2': Normal(0, 0.25),
                'x3': Normal(0, 0.25 ** 2),
            }
        else:
            raise NotImplementedError

        super().__init__(root_dir=root_dir,
                         name=Cte.COLLIDER,
                         eq_type=equations_type,
                         nodes_to_intervene=['x1', 'x2'],
                         structural_eq=structural_eq,
                         noises_distr=noises_distr,
                         adj_edges={'x1': ['x3'],
                                    'x2': ['x3'],
                                    'x3': []},
                         split=split,
                         num_samples=num_samples,
                         likelihood_names=likelihood_names,
                         transform=transform,
                         **kwargs
                         )
