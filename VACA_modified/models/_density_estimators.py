import numpy as np
import torch


def ELBO(log_w):
    return torch.mean(log_w, dim=-1)


def IWAE(log_w, trick=False):
    if not trick:
        return torch.logsumexp(log_w, dim=-1) - np.log(log_w.shape[-1]), {}
    # notice that using the trick is required for computing the gradient wrt the IWAE measure, but doesn't return a
    # lower bound of the log evidence. Therefore use only for computing the gradient. For the estimate,
    # use trick=False

    log_w_max = log_w - torch.max(log_w, dim=-1)[0].view(-1, 1)  # w_k/max(w)
    # normalized_w_k = w_k/(sum_k w_k)
    normalized_w = torch.exp(log_w_max - torch.logsumexp(log_w_max, dim=-1, keepdim=True)).detach().clone()
    info = {}  # {'normalized_w': normalized_w}
    return torch.mul(normalized_w, log_w).sum(-1), info


def IWAE_dreg(log_w, zs):
    # print('IWAE_dreg!')
    with torch.no_grad():
        normalized_w = torch.exp(log_w - torch.logsumexp(log_w, dim=-1, keepdim=True))
        if zs.requires_grad:
            # print('requires grad!')
            zs.register_hook(lambda grad: normalized_w.unsqueeze(-1) * grad)
    info = {}  # {'normalized_w': normalized_w}
    return torch.mul(normalized_w, log_w).sum(-1), info
