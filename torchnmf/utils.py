import torch
import torch.nn.functional as F


def normalize(x: torch.Tensor, axis=0) -> torch.Tensor:
    return x / x.sum(axis, keepdim=True)


def renorm_(input: torch.Tensor, dim=0):
    tmp = input * input
    sum_dims = list(range(input.dim()))
    sum_dims.remove(dim)
    input /= tmp.sum(sum_dims, keepdim=True)
