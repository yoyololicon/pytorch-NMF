import torch
from operator import mul
from functools import reduce
from torch.nn import functional as F
import numpy as np

_fix_neg = torch.nn.Threshold(0, 1e-12)


def KL_divergence(predict, target):
    return (target * _fix_neg(target / _fix_neg(predict)).log()).sum() - target.sum() + predict.sum()


def Euclidean(predict, target):
    return F.mse_loss(predict, target, reduction='sum') / 2


def IS_divergence(predict, target):
    div = target / _fix_neg(predict)
    return div.sum() - _fix_neg(div).log().sum() - reduce(mul, target.shape)


def Beta_divergence(predict, target, beta=2):
    if beta == 2:
        return Euclidean(predict, target)
    elif beta == 1:
        return KL_divergence(predict, target)
    elif beta == 0:
        return IS_divergence(predict, target)
    else:
        bminus = beta - 1
        save_predict, save_target = _fix_neg(predict), _fix_neg(target)
        return (save_target.pow(beta).sum() + bminus * save_predict.pow(beta).sum() - beta * (
                save_target * save_predict.pow(bminus)).sum()) / (beta * bminus)
