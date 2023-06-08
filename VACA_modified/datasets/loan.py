from datasets.toy import ToySCM
from utils.constants import Cte
from utils.distributions import *


class LoanSCM(ToySCM):

    def __init__(self, root_dir,
                 split: str = 'train',
                 num_samples: int = 5000,
                 equations_type=Cte.LINEAR,
                 likelihood_names: str = 'd_d_cb',
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
            e_0 = -1
            e_G = 0.5
            e_A = 1

            l_0 = 1
            l_A = .01
            l_G = 1

            d_0 = -1
            d_A = .1
            d_G = 2
            d_L = 1

            i_0 = -4
            i_A = .1
            i_G = 2
            # i_E = 10
            i_GE = 1

            s_0 = -4
            s_I = 1.5

            structural_eq = {
                # Gender
                'x1': lambda u1,: u1,
                # Age
                'x2': lambda u2,: -35 + u2,
                # Education
                'x3': lambda u3, x1, x2: -0.5 + (
                        1 + np.exp(-(e_0 + e_G * x1 + e_A * (1 + np.exp(- .1 * (x2))) ** (-1) + u3))) ** (-1),
                # Loan amount
                'x4': lambda u4, x1, x2: l_0 + l_A * (x2 - 5) * (5 - x2) + l_G * x1 + u4,
                # Loan duration
                'x5': lambda u5, x1, x2, x4: d_0 + d_A * x2 + d_G * x1 + d_L * x4 + u5,
                # Income
                'x6': lambda u6, x1, x2, x3: i_0 + i_A * (x2 + 35) + i_G * x1 + i_GE * x1 * x3 + u6,
                # Savings
                'x7': lambda u7, x6: s_0 + s_I * (x6 > 0) * x6 + u7,
            }

            noises_distr = {
                # Gender
                'x1': Bernoulli(0.5),
                # Age
                'x2': Gamma(10, 3.5),
                # Education
                'x3': Normal(0, 0.5 ** 2),
                # Loan amount
                'x4': Normal(0, 2 ** 2),
                # Loan duration
                'x5': Normal(0, 3 ** 2),
                # Income
                'x6': Normal(0, 2 ** 2),
                # Savings
                'x7': Normal(0, 5 ** 2),
            }


        else:
            raise NotImplementedError

        super().__init__(root_dir=root_dir,
                         name=Cte.LOAN,
                         eq_type=equations_type,
                         nodes_to_intervene=['x1', 'x2', 'x4', 'x6'],
                         structural_eq=structural_eq,
                         noises_distr=noises_distr,
                         adj_edges={'x1': ['x3', 'x4', 'x5', 'x6'],
                                    'x2': ['x3', 'x4', 'x5', 'x6'],
                                    'x3': ['x6'],
                                    'x4': ['x5'],
                                    'x5': [],
                                    'x6': ['x7'],
                                    'x7': []},
                         split=split,
                         num_samples=num_samples,
                         likelihood_names=likelihood_names,
                         transform=transform,
                         **kwargs
                         )
