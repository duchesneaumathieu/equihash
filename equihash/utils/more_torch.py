import torch
import numpy as np
from functools import reduce

def torch_maximum(*tensors):
    return reduce(torch.max, tensors)

def torch_logsumexp(*tensors):
    maxes = torch_maximum(*tensors)
    maxes = torch.where(maxes==-np.inf, torch.zeros_like(maxes), maxes)
    return sum((tensor - maxes).exp() for tensor in tensors).log() + maxes

def torch_max(tensor, dim=None, keepdims=False):
    if isinstance(dim, int):
        return tensor.max(dim, keepdim=keepdims)[0]
    if dim is None or set(dim)==set(range(tensor.dim())):
        if keepdims: return tensor.max().view(*[1]*tensor.dim())
        else: return tensor.max()
    dim = tuple(d+tensor.dim() if d < 0 else d for d in dim)
    for d in sorted(dim, reverse=True):
        tensor = tensor.max(d, keepdim=keepdims)[0]
    return tensor

def torch_lse(tensor, dim=None, keepdims=False):
    if dim is None: dim = tuple(range(tensor.dim()))
    maxes = torch_max(tensor, dim=dim, keepdims=True)
    maxes = torch.where(maxes==-np.inf, torch.zeros_like(maxes), maxes)
    lse = (tensor - maxes).exp().sum(dim=dim, keepdim=True).log() + maxes
    if not keepdims:
        if isinstance(dim, (tuple,list)):
            dim = tuple(d+tensor.dim() if d < 0 else d for d in dim)
            for d in sorted(dim, reverse=True): lse = lse.squeeze(dim=d)
        else: lse = lse.squeeze(dim=dim)
    return lse