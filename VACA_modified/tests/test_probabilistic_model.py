
# %%
import torch
from utils.probabilistic_model import HeterogeneousDistribution
from utils.likelihoods import get_likelihood
import utils.likelihoods as ul
likelihoods=[ul.NormalLikelihood(1), ul.CategoricalLikelihood(3)]


params_list = [lik.params_size for lik in likelihoods]
logits = torch.randn([16, sum(params_list)])*10


p = HeterogeneousDistribution(likelihoods=likelihoods,
                              norm_categorical=False,
                              norm_by_dim=False)



# %%
p.set_logits(logits)

print(p.mean[:2])


# %%

x = p.sample()

# %%
samples = []

logits_list = torch.split(logits, split_size_or_sections=params_list, dim=1)
for logits_i, lik_i in zip(logits_list, likelihoods):
    distr_i = lik_i(logits_i)
    sample_i = distr_i.sample()
    if  lik_i.name == 'cat':
        y_onehot = torch.FloatTensor(logits_i.shape)
        # In your for loop
        y_onehot.zero_()
        y_onehot.scatter_(1, sample_i.view(-1, 1), 1)
        sample_i = y_onehot
    samples.append(sample_i)

sample = torch.cat(samples, dim=1)

p.log_prob(sample)


# %%
import torch
from utils.probabilistic_model import ProbabilisticModelSCM
likelihoods = []
likelihoods.append([ul.NormalLikelihood(2)]) # First node
likelihoods.append([ul.CategoricalLikelihood(3), ul.CategoricalLikelihood(2)]) # Second node
likelihoods.append([ul.NormalLikelihood(2)]) # Third node

likelihood = ProbabilisticModelSCM(likelihoods=likelihoods,
                                   embedding_size=8,
                                   act_name='relu',
                                   drop_rate=0.0,
                                   norm_categorical=False,
                                   norm_by_dim=False)



num_graphs = 3
logits = torch.randn([num_graphs*likelihood.num_nodes, likelihood.embedding_size])


px = likelihood(logits, return_mean=False)


# %% Test distributions
from utils.distributions import *
Normal(0,1).sample()
Normal(0,1).sample(10)
Normal(0,1).pdf(0)
Normal(0,1).visualize()

MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]).sample()
MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]).sample(10)
MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]).pdf(0)
MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]).pdf(+2)
MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]).pdf(-2)
MixtureOfGaussians([0.5, 0.5], [-2, +2], [1, 1]).visualize()