from datasets.toy import ToySCM
from utils.constants import Cte
from utils.distributions import *


class MGraphSCM(ToySCM):

    def __init__(self, root_dir,
                 split: str = 'train',
                 num_samples: int = 5000,
                 equations_type=Cte.LINEAR,
                 likelihood_names: str = 'd_d_d_d_d',
                 transform=None,
                 **kwargs
                 ):
        """
        Args:
            root_dir: path to data directory
            train: whether to load the training subset (``True``, ``'train-*'`` files) or the test
                subset (``False``, ``'t10k-*'`` files)
            columns: list of morphometrics to load; by default (``None``) loads the image index and
                all available metrics: area, length, thickness, slant, width, and height
        """

        if equations_type == Cte.LINEAR:
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2: u2,
                'x3': lambda u3, x1: x1 + u3,
                'x4': lambda u4, x1, x2: -x2 + 0.5 * x1 + u4,
                'x5': lambda u5, x2: -1.5 * x2 + u5,
            }

            noises_distr = {
                'x1': Normal(0, 1),
                'x2': Normal(0, 1),
                'x3': Normal(0, 1),
                'x4': Normal(0, 1),
                'x5': Normal(0, 1),
            }


        elif equations_type == Cte.NONLINEAR:
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2: u2,
                'x3': lambda u3, x1: x1 + 0.5 * x1 ** 2 + u3,
                'x4': lambda u4, x1, x2: -x2 + 0.5 * x1 ** 2 + u4,
                'x5': lambda u5, x2: -1.5 * x2 ** 2 + u5,
            }

            noises_distr = {
                'x1': Normal(0, 1),
                'x2': Normal(0, 1),
                'x3': Normal(0, 1),
                'x4': Normal(0, 1),
                'x5': Normal(0, 1),
            }

        elif equations_type == Cte.NONADDITIVE:
            structural_eq = {
                'x1': lambda u1: u1,
                'x2': lambda u2: u2,
                'x3': lambda u3, x1: (x1) * u3,
                'x4': lambda u4, x1, x2: (-x2 + 0.5 * x1 ** 2) * u4,
                'x5': lambda u5, x2: (-1.5 * x2 ** 2) * u5,
            }

            noises_distr = {
                'x1': Normal(0, 1),  # MixtureOfGaussians(probs=[0.5, 0.5], means=[-2.5, 2.5], vars=[1, 1]),
                'x2': Normal(0, 1),
                'x3': Normal(0, 1),
                'x4': Normal(0, 1),
                'x5': Normal(0, 1),
            }

        else:
            raise NotImplementedError

        super().__init__(root_dir=root_dir,
                         name=Cte.MGRAPH,
                         eq_type=equations_type,
                         nodes_to_intervene=['x1', 'x2'],
                         structural_eq=structural_eq,
                         noises_distr=noises_distr,
                         adj_edges={'x1': ['x3', 'x4'],
                                    'x2': ['x4', 'x5'],
                                    'x3': [],
                                    'x4': [],
                                    'x5': []},
                         split=split,
                         num_samples=num_samples,
                         likelihood_names=likelihood_names,
                         transform=transform,
                         **kwargs
                         )
