import torch
from torch.nn import functional as F

eps = 1e-8


def KL_divergence(V_tilde, V):
    """
    The generalized Kullback-Leibler divergence.

    :param V: Target matrix.
    :param V_tilde: Reconstructed matrix.
    :return: Distance.
    """
    return torch.sum(V * torch.log(V / V_tilde + eps) - V + V_tilde)


def Euclidean(V_tilde, V):
    """
    Squared Frobenius norm.

    :param V: Target matrix.
    :param V_tilde: Reconstructed matrix.
    :return: Distance.
    """
    return F.mse_loss(V_tilde, V, reduction='sum') / 2


def IS_divergence(V_tilde, V):
    """
    The Itakura-Saito divergence:

    :param V: Target matrix.
    :param V_tilde: Reconstructed matrix.
    :return: Distance.
    """
    VV = V / V_tilde
    return torch.sum(VV - torch.log(VV + eps) - 1)


def Beta_divergence(V_tilde, V, beta=2.):
    if beta == 1:
        return KL_divergence(V_tilde, V)
    elif beta == 0:
        return IS_divergence(V_tilde, V)
    else:
        bminus = beta - 1
        return torch.sum((V ** beta + bminus * V_tilde ** beta - beta * V * V_tilde ** bminus) / (beta * bminus))


if __name__ == '__main__':
    a = torch.rand(10)
    b = torch.rand(10)

    print(Beta_divergence(a, b).item(), Euclidean(a, b).item())
    print(Beta_divergence(a, b, 1).item(), KL_divergence(a, b).item())
    print(Beta_divergence(a, b, 0).item(), IS_divergence(a, b).item())