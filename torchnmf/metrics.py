import torch
from operator import mul
from functools import reduce
from torch.nn import functional as F

eps = 1e-8


def KL_divergence(predict, target):
    """
    The generalized Kullback-Leibler divergence.

    :param V: Target matrix.
    :param V_tilde: Reconstructed matrix.
    :return: Distance.
    """
    return torch.sum(target * torch.log(target / predict + eps)) - target.sum() + predict.sum()


def Euclidean(predict, target):
    """
    Squared Frobenius norm.

    :param V: Target matrix.
    :param V_tilde: Reconstructed matrix.
    :return: Distance.
    """
    return F.mse_loss(predict, target, reduction='sum') / 2


def IS_divergence(predict, target):
    """
    The Itakura-Saito divergence:

    :param V: Target matrix.
    :param V_tilde: Reconstructed matrix.
    :return: Distance.
    """
    div = target / predict
    return div.sum() - torch.log(div + eps).sum() - reduce(mul, target.shape)


def Beta_divergence(predict, target, beta=2.):
    if beta == 2:
        return Euclidean(predict, target)
    elif beta == 1:
        return KL_divergence(predict, target)
    elif beta == 0:
        return IS_divergence(predict, target)
    else:
        bminus = beta - 1
        return (torch.sum(target ** beta) + bminus * torch.sum(predict ** beta) - beta * torch.sum(
            target * predict ** bminus)) / (beta * bminus)


if __name__ == '__main__':
    a = torch.rand(10)
    b = torch.rand(10)

    print(Beta_divergence(a, b).item(), Euclidean(a, b).item())
    print(Beta_divergence(a, b, 1).item(), KL_divergence(a, b).item())
    print(Beta_divergence(a, b, 0).item(), IS_divergence(a, b).item())
