import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import Cte


def get_likelihood(name, domain_size):
    lik_cls = None
    if name == Cte.BETA:  # Beta
        lik_cls = BetaLikelihood
    elif name == Cte.BERNOULLI:  # Bernoulli
        lik_cls = BernoulliLikelihood
    elif name == Cte.CATEGORICAL:  # Categorical
        lik_cls = CategoricalLikelihood
    elif name == Cte.CONTINOUS_BERN:  # Continuous Bernoulli
        lik_cls = ContinousBernoulliLikelihood
    elif name == Cte.DELTA:  # Delta
        lik_cls = DeltaLikelihood
    elif name == Cte.GAUSSIAN:
        lik_cls = NormalLikelihood
    elif name == 'normal_fix':
        lik_cls = NormalLikelihoodFixStd
    else:
        raise NotImplementedError()

    return lik_cls(domain_size)


class BaseLikelihood(nn.Module):
    def __init__(self,
                 domain_size: int,
                 normalize: str):
        """
        Base class to define the likelihoods
        Args:
            domain_size:
            normalize:
            String identifying the type of normalization. It can take three possible values
                all: normalize all dimensions equally
                dim: normalize per dimension
                otherwise: do not normalize
        """
        super(BaseLikelihood, self).__init__()
        self._domain_size = domain_size
        self._scalar_params = None
        self.normalize = normalize

    @property
    def name(self):
        raise NotImplementedError

    @property
    def domain_size(self):
        return self._domain_size

    @property
    def params_size(self):
        raise NotImplementedError

    def fit(self, x):
        raise NotImplementedError

    def fit_normalize_data(self, x):
        self.fit(x)
        return self.normalize_data(x)

    def normalize_data(self, x):
        raise NotImplementedError

    def denormalize_data(self, x_norm):
        raise NotImplementedError

    def denormalize_params(self, params):
        raise NotImplementedError

    def has_fit(self, include_size=False):
        raise NotImplementedError


class BetaLikelihood(BaseLikelihood):
    def __init__(self, domain_size):
        super().__init__(domain_size)

    @property
    def params_size(self):
        return self._domain_size * 2

    def forward(self, logits, return_mean=False):
        logits = F.softplus(logits)
        latent_dim = logits.size(1) // 2
        c0, c1 = torch.split(logits, split_size_or_sections=latent_dim, dim=1)
        p = td.Beta(c0, c1)
        if return_mean:
            return p.mean, p
        else:
            return p


class BernoulliLikelihood(BaseLikelihood):
    def __init__(self, domain_size, normalize='all'):
        super().__init__(domain_size, normalize)

    @property
    def name(self):
        return Cte.BERNOULLI

    @property
    def params_size(self):
        return self._domain_size

    def forward(self, logits, return_mean=False):
        p = td.Bernoulli(logits=logits)
        if return_mean:
            return p.mean, p
        else:
            return p

    def fit(self, x):
        return

    def normalize_data(self, x):
        return x

    def fit_normalize_data(self, x):
        self.fit(x)
        return self.normalize_data(x)

    def denormalize_data(self, x_norm):
        return x_norm

    def denormalize_params(self, p):
        return

    def has_fit(self, include_size=False):
        if include_size:
            return [False, ] * self.domain_size
        else:
            return False


class CategoricalLikelihood(BaseLikelihood):
    def __init__(self, domain_size, normalize='all'):
        super().__init__(domain_size, normalize)

    @property
    def name(self):
        return Cte.CATEGORICAL

    @property
    def params_size(self):
        return self._domain_size

    def forward(self, logits, return_mean=False):
        p = td.Categorical(logits=logits)
        if return_mean:
            return F.softmax(logits), p
        else:
            return p

    def fit(self, x):
        return

    def normalize_data(self, x):
        return x

    def fit_normalize_data(self, x):
        self.fit(x)
        return self.normalize_data(x)

    def denormalize_data(self, x_norm):
        return x_norm

    def denormalize_params(self, p):
        return

    def has_fit(self, include_size=False):
        if include_size:
            return [False, ] * self.domain_size
        else:
            return False


class ContinousBernoulliLikelihood(BaseLikelihood):
    def __init__(self, domain_size, normalize='all'):
        super().__init__(domain_size, normalize)

    @property
    def name(self):
        return Cte.CONTINOUS_BERN

    @property
    def params_size(self):
        return self._domain_size

    def forward(self, logits, return_mean=False):
        p = td.ContinuousBernoulli(logits=logits)

        if return_mean:
            return F.sigmoid(logits), p
        else:
            return p

    def fit(self, x):
        assert x.shape[1] == self._domain_size
        self._scalar_params = {}
        if self.normalize == 'all':
            self._scalar_params['max'] = x.max()
            self._scalar_params['min'] = x.min()
        elif self.normalize == 'dim':
            self._scalar_params['max'] = x.max(0)
            self._scalar_params['min'] = x.min(0)
        else:
            self._scalar_params['max'] = 1.
            self._scalar_params['min'] = 0.

    def normalize_data(self, x):
        max_ = self._scalar_params['max']
        min_ = self._scalar_params['min']
        x_norm = (x - min_) / (max_ - min_)
        return x_norm

    def denormalize_data(self, x_norm):
        max_ = self._scalar_params['max']
        min_ = self._scalar_params['min']
        x = x_norm * (max_ - min_) + min_
        return x

    def denormalize_params(self, p):
        raise NotImplementedError

    def has_fit(self, include_size=False):
        if include_size:
            return [True, ] * self.domain_size
        else:
            return True


class DeltaLikelihood(BaseLikelihood):
    def __init__(self, domain_size, normalize='dim', lambda_=1.0):
        super().__init__(domain_size, normalize)

        self.lambda_ = lambda_

    @property
    def name(self):
        return Cte.DELTA

    @property
    def params_size(self):
        return self._domain_size

    def set_lambda(self, value):
        self.lambda_ = value

    def forward(self, logits, return_mean=False):
        # logits = torch.sigmoid(logits)
        p = Delta(logits, lambda_=self.lambda_)
        if return_mean:
            return p.mean, p
        else:
            return p

    def fit(self, x):
        assert x.shape[1] == self._domain_size
        self._scalar_params = {}
        if self.normalize == 'all':

            self._scalar_params['mu'] = x.mean()
            self._scalar_params['std'] = x.std()
        elif self.normalize == 'dim':
            std = x.std(0)
            std[std == 0] = 1.
            self._scalar_params['mu'] = x.mean(0)
            self._scalar_params['std'] = std
        else:
            self._scalar_params['mu'] = 0.
            self._scalar_params['std'] = 1.

    def normalize_data(self, x):
        mu = self._scalar_params['mu']
        std = self._scalar_params['std']
        x_norm = (x - mu) / std
        return x_norm

    def denormalize_data(self, x_norm):
        mu = self._scalar_params['mu']
        std = self._scalar_params['std']
        if x_norm.get_device()!= mu.get_device():
            x_norm = x_norm.cpu()

        x = x_norm * std + mu
        return x

    def denormalize_params(self, p):
        mu = self._scalar_params['mu']
        std = self._scalar_params['std']

        p.loc = p.loc * std + mu
        p.scale = p.scale * std

    def has_fit(self, include_size=False):
        if include_size:
            return [True, ] * self.domain_size
        else:
            return True


class NormalLikelihood(BaseLikelihood):
    def __init__(self, domain_size, normalize='dim'):
        super().__init__(domain_size, normalize)
        self.clip_std = 0.0
        self.fix_std = None

    @property
    def name(self):
        return Cte.GAUSSIAN

    def set_fix_std(self, value):
        self.fix_std = value

    @property
    def params_size(self):
        return self._domain_size * 2

    def forward(self, logits, return_mean=False):
        latent_dim = logits.size(1) // 2
        mu, log_var = torch.split(logits, split_size_or_sections=latent_dim, dim=1)
        std = torch.exp(log_var / 2) + 0.0001
        if self.clip_std > 0:
            std = torch.clip(std, max=self.clip_std)
        if isinstance(self.fix_std, float):
            std = torch.ones_like(std, requires_grad=False) * self.fix_std
        # std = 0.001*torch.sigmoid(log_var)
        p = td.Normal(mu, std)
        if return_mean:
            return mu, p
        else:
            return p

    def fit(self, x):
        assert x.shape[1] == self._domain_size
        self._scalar_params = {}
        if self.normalize == 'all':

            self._scalar_params['mu'] = x.mean()
            self._scalar_params['std'] = x.std()
        elif self.normalize == 'dim':
            std = x.std(0)
            std[std == 0] = 1.
            self._scalar_params['mu'] = x.mean(0)
            self._scalar_params['std'] = std
        else:
            self._scalar_params['mu'] = 0.
            self._scalar_params['std'] = 1.

    def normalize_data(self, x):
        mu = self._scalar_params['mu']
        std = self._scalar_params['std']
        x_norm = (x - mu) / std
        return x_norm

    def denormalize_data(self, x_norm):
        mu = self._scalar_params['mu']
        std = self._scalar_params['std']
        x = x_norm * std + mu
        return x

    def denormalize_params(self, p):
        mu = self._scalar_params['mu']
        std = self._scalar_params['std']

        p.loc = p.loc * std + mu
        p.scale = p.scale * std

    def has_fit(self, include_size=False):
        if include_size:
            return [True, ] * self.domain_size
        else:
            return True


class NormalLikelihoodFixStd(BaseLikelihood):
    def __init__(self, domain_size):
        super().__init__(domain_size)

        self.fix_std = 0.01

    @property
    def params_size(self):
        return self._domain_size

    def forward(self, logits, return_mean=False):

        std = torch.ones_like(logits, requires_grad=False) * self.fix_std
        # std = 0.001*torch.sigmoid(log_var)

        p = td.Normal(logits, std)
        if return_mean:
            return logits, p
        else:
            return p


# %%

class Delta(td.Distribution):
    def __init__(self, center=None, lambda_=1.0, validate_args=None):
        if center is None:
            raise ValueError("`center` must be specified.")
        self.center = center
        self.lambda_ = lambda_
        self._param = self.center
        batch_shape = self._param.size()
        super(Delta, self).__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        return self.center

    def sample(self, sample_shape=torch.Size()):
        return self.center

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError()

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        return - (1 / self.lambda_) * (value - self.center) ** 2
