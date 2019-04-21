import torch
from operator import mul
from functools import reduce
from torch.nn import functional as F

eps = 1e-8


def KL_divergence(predict, target):
    return (target * (target / predict).log()).sum() - target.sum() + predict.sum()


def Euclidean(predict, target):
    return F.mse_loss(predict, target, reduction='sum') / 2


def IS_divergence(predict, target):
    div = target / predict
    return div.sum() - div.log().sum() - reduce(mul, target.shape)


def Beta_divergence(predict, target, beta=2):
    if beta == 2:
        return Euclidean(predict, target)
    elif beta == 1:
        return KL_divergence(predict, target)
    elif beta == 0:
        return IS_divergence(predict, target)
    else:
        bminus = beta - 1
        return (target.pow(beta).sum() + bminus * predict.pow(beta).sum() - beta * (
                target * predict.pow(bminus)).sum()) / (beta * bminus)
