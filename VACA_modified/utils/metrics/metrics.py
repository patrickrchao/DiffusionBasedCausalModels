import torch
from torch.distributions import Normal


def accuracy(preds, targets, one_hot_preds=True, dim=-1):
    if one_hot_preds:
        preds = preds.argmax(dim=dim)

    return (preds == targets).sum().float() / len(targets)


def kl_divergence_normal(mu1, log_std1, mu2=None, log_std2=None):
    q = Normal(mu1, torch.exp(log_std1))
    if mu2 == None or log_std2 == None:
        mu2 = torch.zeros_like(mu1)
        log_std2 = torch.zeros_like(log_std1)
    p = Normal(mu2, torch.exp(log_std2))

    return torch.distributions.kl_divergence(q, p)
