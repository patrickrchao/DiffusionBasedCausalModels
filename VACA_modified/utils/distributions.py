import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import bernoulli
from scipy.stats import norm  # norm for univariate; use multivariate_normal otherwise


# univariate distributions
class BaseDistribution(object):
    def __init__(self):
        pass

    def sample(self, size=1):
        raise NotImplementedError

    def pdf(self):
        raise NotImplementedError

    def visualize(self):
        plt.hist(self.sample(500), 50, facecolor='green', alpha=0.75)
        plt.ylabel('Count')
        plt.title(fr'Histogram of {self.name}')
        plt.grid(True)
        plt.show()


class Normal(BaseDistribution):
    def __init__(self, mean, var):
        assert isinstance(mean, int) or isinstance(mean, float), 'Expected `mean` to be an int or float.'
        assert isinstance(var, int) or isinstance(var, float), 'Expected `var` to be an int or float.'
        self.mean = mean
        self.var = var
        self.name = f'Normal\t mean={self.mean}, var={self.var}'

    def sample(self, size=1):
        tmp = [np.random.normal(self.mean, np.sqrt(self.var)) for _ in range(size)]
        return tmp[0] if size == 1 else tmp

    def pdf(self, value):
        return norm(self.mean, self.var).pdf(value)


class MixtureOfGaussians(BaseDistribution):

    def __init__(self, probs, means, vars):
        assert sum(probs) == 1, 'Mixture probabilities must sum to 1.'
        assert len(probs) == len(means) == len(vars), 'Length mismatch.'
        self.probs = probs
        self.means = means
        self.vars = vars
        self.name = f'MoG\t probs={self.probs}, means={self.means}, vars={self.vars}'

    def sample(self, size=1):
        tmp = [
            np.random.normal(self.means[mixture_idx], np.sqrt(self.vars[mixture_idx]))
            for mixture_idx in np.random.choice(len(self.probs), size=size, p=self.probs)
        ]
        return tmp[0] if size == 1 else tmp

    def pdf(self, value):
        return np.sum([
            prob * norm(mean, var).pdf(value)
            for (prob, mean, var) in zip(self.probs, self.means, self.vars)
        ])


class Bernoulli(BaseDistribution):

    def __init__(self, prob):
        assert isinstance(prob, int) or isinstance(prob, float), 'Expected `prob` to be an int or float.'
        assert prob >= 0 and prob <= 1

        self.prob = prob
        self.name = f'Bernoulli\t prob={self.prob}'

    def sample(self, size=1):
        tmp = bernoulli.rvs(self.prob, size=size)
        return tmp[0] if size == 1 else list(tmp)

    def pdf(self, value):
        raise Exception(f'not supported yet; code should not come here.')


class Poisson(BaseDistribution):

    def __init__(self, p_lambda):
        assert isinstance(p_lambda, int) or isinstance(p_lambda, float), 'Expected `p_lambda` to be an int or float.'
        assert p_lambda > 0
        self.p_lambda = p_lambda
        self.name = f'Poisson\t prob={self.p_lambda}'

    def sample(self, size=1):
        tmp = np.random.poisson(self.p_lambda, size)
        return tmp[0] if size == 1 else list(tmp)

    def pdf(self, value):
        raise Exception(f'not supported yet; code should not come here.')


class Gamma(BaseDistribution):

    def __init__(self, shape, scale):
        assert isinstance(shape, int) or isinstance(shape, float), 'Expected `shape` to be an int or float.'
        assert isinstance(scale, int) or isinstance(scale, float), 'Expected `scale` to be an int or float.'
        assert shape > 0
        assert scale > 0
        self.shape = shape
        self.scale = scale
        self.name = f'Gamma\t shape={self.shape}, scale={self.scale}'

    def sample(self, size=1):
        tmp = np.random.gamma(self.shape, self.scale, size)
        return tmp[0] if size == 1 else list(tmp)

    def pdf(self, value):
        raise Exception(f'not supported yet; code should not come here.')


class Categorical(BaseDistribution):

    def __init__(self, probs):
        assert isinstance(probs, list), 'Expected `probs` to be a list.'
        assert np.sum(probs) == 1
        assert np.sum(probs == 0) == 0
        self.probs = probs
        self.num_categories = len(probs)
        self.name = f'Categorical\t probs={self.probs}'

    def sample(self, size=1):
        tmp = np.random.choice(self.num_categories, size=size, p=self.probs)
        return tmp[0] if size == 1 else list(tmp)

    def pdf(self, value):
        raise Exception(f'not supported yet; code should not come here.')


class Laplace(BaseDistribution):

    def __init__(self, loc, scale):
        assert isinstance(loc, int) or isinstance(loc, float), 'Expected `shape` to be an int or float.'
        assert isinstance(scale, int) or isinstance(scale, float), 'Expected `scale` to be an int or float.'
        assert loc >= 0
        assert scale > 0
        self.loc = loc
        self.scale = scale
        self.name = f'Laplace\t loc={self.loc}, scale={self.scale}'

    def sample(self, size=1):
        tmp = np.random.laplace(self.loc, self.scale, size)
        return tmp[0] if size == 1 else list(tmp)

    def pdf(self, value):
        raise Exception(f'not supported yet; code should not come here.')


class Empirical(BaseDistribution):

    def __init__(self, data):
        if len(data.shape) >1 :
            data = data.flatten()
        self.data = data
        
        self.name = f'Empirical'

    def sample(self, size=1):
        return np.random.choice(self.data, size,replace=True)

    def pdf(self, value):
        raise Exception(f'not supported yet; code should not come here.')
