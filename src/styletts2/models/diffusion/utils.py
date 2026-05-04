import torch


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def rand_bool(shape, proba, device=None):
    if proba == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    if proba == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    return torch.bernoulli(torch.full(shape, proba, device=device)).to(torch.bool)
