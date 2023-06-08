import numpy as np
from scipy import stats

from datasets.toy import ToySCM
from utils.constants import Cte
from utils.distributions import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


RACE = 'race'
AGE = 'age'
NATIVE_COUNTRY = 'native_country'
GENDER = 'gender'

EDU = 'edu'
HOUR = 'hour'
WORK_CLASS = 'work_class'
MARITIAL = 'maritial'

OCCUPATION = 'occupation'
RELATIONSHIP = 'relationship'
INCOME = 'income'


class AdultSCM(ToySCM):

    def __init__(self, root_dir,
                 split: str = 'train',
                 num_samples: int = 5000,
                 equations_type=Cte.LINEAR,
                 likelihood_names: str = 'c_d_c_b_d_d_c_c_c_c_d',
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

            def eq_race(u):
                '''
                Three categories
                '''
                return u.copy()

            def eq_age(u):
                return u + 17

            def eq_native_country(u):
                return u.copy()

            def eq_gender(u):
                return u.copy()

            def eq_edu(u, race, age, native_country, gender):
                '''
                race: [N, 3]
                age: [N, 1]
                native_country: [N, 4]
                gender: [N, 1]
                '''
                r = 2 * race[:, [0]] + 1 * race[:, [1]] + 0 * race[:, [2]]
                a = sigmoid(age - 30)
                n = 0 * native_country[:, [0]] + 2 * native_country[:, [1]] + 5 * native_country[:,
                                                                                  [2]] + 1 * native_country[:, [3]]
                g = 0.5 * (gender == 0) + 1.0 * (gender == 1)

                return np.exp(r + a) + g * n + u

            def eq_hour(u, race, edu, age, native_country, gender):
                '''
                race: [N, 3]
                edu: [N, 1]
                age: [N, 1]
                native_country: [N, 4]
                gender: [N, 1]
                '''
                r = 0.5 * race[:, [0]] + 1 * race[:, [1]] + 1.3 * race[:, [2]]
                e = 5 * np.abs(np.tanh(edu - 2))  #
                a = 2 * np.exp(-(age - 30) ** 2)
                n = 40 * native_country[:, [0]] + 36 * native_country[:, [1]] + 50 * native_country[:,
                                                                                     [2]] + 30 * native_country[:, [3]]
                g = 2 * (gender == 0) + 0 * (gender == 1)
                out = n * r + a + e + g

                return (out + u) * (age < 70)

            def eq_work_class(u, age, edu, native_country, hour):
                '''
                edu: [N, 1]
                age: [N, 1]
                native_country: [N, 4]
                hour: [N, 1]
                '''
                a = age + 1.5 * u
                e = 5 * np.abs(np.tanh(edu - 2))  #
                h = sigmoid(hour - 30 + u)
                n = -1 * native_country[:, [0]] + 0 * native_country[:, [1]] + 1 * native_country[:,
                                                                                   [2]] + 2 * native_country[:, [3]]

                out = 0 * (a < 17) + 1 * ((e + h) > 0.3) + 1 * (h > 0.3) * (a > 50) + n
                out[out > 3] = 3.
                return out * (out >= 0)

            def eq_maritial(u, race, age, work_class, hour, native_country, gender):
                '''
                Threee categories
                  race: [N, 3]
                  age: [N, 1]
                  work_class: [N, 4]
                  hour: [N, 1]
                  native_country: [N, 4]
                  gender: [N, 1]
                  '''
                r = race.copy().argmax(1).reshape(-1, 1)
                r = (r + u * 0.2).astype(np.int32)

                r[r < 0] = 0
                r[r > 2] = 2
                r = 0 * (r == 0) + 2 * (r == 1) + 1 * (r == 2)

                a = age + u * 2
                a = 0 * (a < 20) + 2 * (a >= 20) & (a < 40) + 1 * (a >= 40) & (a < 50) + 2 * (a >= 50)

                w = 1 * work_class[:, [0]] + 1 * work_class[:, [1]] + 0 * work_class[:, [2]] + 2 * work_class[:, [3]]
                h = 3 * (sigmoid(hour - 30)).astype(int)
                h[h > 2] = 2

                n = 0 * native_country[:, [0]] + 1 * native_country[:, [1]] + 1 * native_country[:,
                                                                                  [2]] + 2 * native_country[:, [3]]

                g = (gender + u * 0.5).astype(int)

                g[g < 0] = 0
                g[g > 1] = 1
                g = 1 * (g == 0) + 2 * (g == 1)

                out = np.concatenate([r, a, w, h, n, g], axis=1)
                out = stats.mode(out, axis=1)[0]

                return out

            def eq_occupation(u, race, age, edu, work_class, maritial, gender):
                '''Three categories
                   race: [N, 3]
                   age: [N, 1]
                   edu: [N, 1]
                   work_class: [N, 4]
                   maritial: [N, 3]
                   gender: [N, 1]
                '''
                comb = race.copy().argmax(1).reshape(-1, 1).astype(np.float32)
                comb += 2 * np.exp(-(age + u - 20) ** 2)
                comb -= sigmoid(edu * u - 30)
                comb += work_class.copy().argmax(1).reshape(-1, 1)
                comb += 3 * maritial.copy().argmax(1).reshape(-1, 1)
                comb += 4 * gender

                out = 0 * (comb < 1)

                out += 1 * (comb >= 1) & (comb <= 4)

                out += 2 * (comb > 4)

                # out = 0*(comb<1) + 1*(comb>=1)&(comb<=4) + 2*(comb>4)

                return out

            def eq_relationship(u, maritial, edu, age, native_country, gender):
                '''Three categories: wife(0), husband(1), not-in-family(2)
                   maritial: [N, 3]
                   age: [N, 1]
                   edu: [N, 1]
                   native_country: [N, 4]
                   gender: [N, 1]
                '''

                comb = u * native_country[:, [0]] - u * native_country[:, [1]] + 2 * u * native_country[:,
                                                                                         [2]] + 2 * native_country[:,
                                                                                                    [3]]

                comb += sigmoid(edu - 30)
                comb += 2 * (age < 20)

                comb += -2 * (gender == 0)

                out = comb.copy()
                m = maritial.copy().argmax(1).reshape(-1, 1).astype(np.float32)
                out[(m == 1) & (comb < -1)] = 0
                out[(m == 1) & (comb >= -1)] = 1
                out[(m != 1) & (comb >= -1)] = 2
                out[(m != 1) & (comb < -1)] = 1
                return out

            def eq_income(u, race, age, edu,
                          occupation, work_class, maritial,
                          hour, native_country, gender, relationship):
                '''
                race: [N, 3]
                age: [N, 1]
                edu: [N, 1]
                occupation: [N, 1]
                work_class: [N, 1]
                maritial: [N, 1]
                hour: [N, 1]
                native_country: [N, 1]
                gender: [N, 1]
                relationship: [N, 1]
                '''
                output = u.copy()
                r = race.copy().argmax(1).reshape(-1, 1).astype(np.float32)
                output += 10000 * (r > 1.6) + 20000 * (r <= 1.6)

                output += 3000 * (age >= 21) & (age < 30) + 8000 * (age >= 30)
                output += 5000 * (edu < 2) + 10000 * (edu >= 2) & (edu < 10) + 30000 * (edu >= 10)

                output += 5000 * occupation[:, [1]] + 15000 * occupation[:, [2]]
                output += 5000 * work_class[:, [0]] + 7000 * work_class[:, [1]]
                output += 1000 * maritial[:, [0]] + 4000 * maritial[:, [1]] - 2000 * maritial[:, [2]]
                output += 15000 * (hour > 45)

                n = native_country.copy().argmax(1).reshape(-1, 1).astype(np.float32)
                output += 10000 * (n > 2)
                output += 4000 * (gender == 1)
                rel = relationship.copy().argmax(1).reshape(-1, 1).astype(np.float32)
                output += 3000 * (rel < 2)

                return output

            structural_eq = {
                RACE: eq_race,
                AGE: eq_age,
                NATIVE_COUNTRY: eq_native_country,
                GENDER: eq_gender,
                EDU: eq_edu,
                HOUR: eq_hour,
                WORK_CLASS: eq_work_class,
                MARITIAL: eq_maritial,
                OCCUPATION: eq_occupation,
                RELATIONSHIP: eq_relationship,
                INCOME: eq_income,
            }

            noises_distr = {
                RACE: Categorical([0.85, 0.1, 0.05]),  # Discrete, 3 categories
                AGE: Gamma(3, 10),  # Continuous, positive
                NATIVE_COUNTRY: Categorical([0.3, 0.5, 0.1, 0.1]),  # Discrete, 4 categories
                GENDER: Bernoulli(0.67),  # Discrete, 2 categories
                EDU: Gamma(1, 1),  # Continuous, positive
                HOUR: Normal(0, 1),  # Continuous, positive
                WORK_CLASS: Normal(0, 1),  # Discrete, 4 categories
                MARITIAL: Normal(0, 1),  # Discrete, 3 categories
                OCCUPATION: MixtureOfGaussians(probs=[0.5, 0.5], means=[-2.5, 2.5], vars=[1, 1]),
                # Discrete, 3 categories
                RELATIONSHIP: Normal(0, 1),  # Discrete, 3 categories
                INCOME: Gamma(1, 1),  # Continuous, positive
            }


        else:
            raise NotImplementedError

        adj_edges = {RACE: [EDU, HOUR, MARITIAL, OCCUPATION, INCOME],
                     AGE: [INCOME, OCCUPATION, MARITIAL, WORK_CLASS, EDU, HOUR, RELATIONSHIP],
                     NATIVE_COUNTRY: [EDU, HOUR, MARITIAL, RELATIONSHIP, INCOME, WORK_CLASS],
                     GENDER: [EDU, HOUR, MARITIAL, OCCUPATION, RELATIONSHIP, INCOME],
                     EDU: [INCOME, OCCUPATION, WORK_CLASS, HOUR, RELATIONSHIP],
                     HOUR: [WORK_CLASS, MARITIAL, INCOME],
                     WORK_CLASS: [OCCUPATION, INCOME, MARITIAL],
                     MARITIAL: [OCCUPATION, INCOME, RELATIONSHIP],
                     OCCUPATION: [INCOME],
                     RELATIONSHIP: [INCOME],
                     INCOME: []}

        nodes_list = list(adj_edges.keys())

        self.categorical_nodes = [RACE, MARITIAL, RELATIONSHIP, NATIVE_COUNTRY, WORK_CLASS, OCCUPATION]

        super().__init__(root_dir=root_dir,
                         name=Cte.ADULT,
                         eq_type=equations_type,
                         nodes_to_intervene=[AGE, EDU, HOUR],
                         structural_eq=structural_eq,
                         noises_distr=noises_distr,
                         adj_edges=adj_edges,
                         split=split,
                         num_samples=num_samples,
                         likelihood_names=likelihood_names,
                         transform=transform,
                         nodes_list=nodes_list,
                         **kwargs
                         )

    @property
    def likelihoods(self):
        likelihoods_tmp = []

        for i, lik_name in enumerate(self.likelihood_names):  # Iterate over nodes
            if self.nodes_list[i] in [RACE, MARITIAL, RELATIONSHIP, OCCUPATION]:
                dim = 3
            elif self.nodes_list[i] in [NATIVE_COUNTRY, WORK_CLASS]:
                dim = 4
            else:
                dim = 1
            likelihoods_tmp.append([self._get_lik(lik_name,
                                                  dim=dim,
                                                  normalize='dim')])

        return likelihoods_tmp

    def sample_obs(self, obs_id, parents_dict=None, n_samples=1, u=None):
        '''
        Only possible if the true Structural Eauations are known
        f = self.structural_eq[f'x{obs_id}']
        if u is None:
            u = np.array(self.noises_distr[f'u{obs_id}'].sample(n_samples))

        if not isinstance(parents_dict, dict):
            return f(u), u
        else:
            return f(u, **parents_dict), u
        '''
        assert obs_id < len(self.nodes_list)
        node_name = self.nodes_list[obs_id]
        lik = self.likelihoods[obs_id][0]
        f = self.structural_eq[node_name]
        u_is_none = u is None
        if u_is_none:
            u = self._sample_noise(node_name, n_samples)
        x = f(u, **parents_dict) if isinstance(parents_dict, dict) else f(u)
        x = x.astype(np.float32)
        if node_name in self.categorical_nodes:
            x_out = np.zeros([x.shape[0], lik.domain_size])
            x = x.astype(np.int32)
            for i in range(x.shape[0]):
                x_out[i, x[i, 0]] = 1

            x = x_out.copy().astype(np.float32)

        return x, u
