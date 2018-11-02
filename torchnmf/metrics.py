import torch
import numpy as np


def KL_divergence(V, V_tilde):
    """
    The generalized Kullback-Leibler divergence.

    :param V: Target matrix.
    :param V_tilde: Reconstructed matrix.
    :return: Distance.
    """
    if type(V) != type(V_tilde):
        raise TypeError

    if type(V) == np.ndarray:
        idx = V.nonzero()
        log = np.log
    elif type(V) == torch.Tensor:
        id1, *id2 = V.nonzero().t()
        idx = [id1] + id2
        log = torch.log
    else:
        raise TypeError
    nonzeroV = V[idx]
    return (nonzeroV * log(nonzeroV / V_tilde[idx])).sum() + (V_tilde - V).sum()


def Frobenius(V, V_tilde):
    """
    Squared Frobenius norm.

    :param V: Target matrix.
    :param V_tilde: Reconstructed matrix.
    :return: Distance.
    """
    return (V - V_tilde).pow(2).sum() / 2


def IS_divergence(V, V_tilde):
    """
    The Itakura-Saito divergence:

    :param V: Target matrix.
    :param V_tilde: Reconstructed matrix.
    :return: Distance.
    """
    if type(V) != type(V_tilde):
        raise TypeError

    if type(V) == np.ndarray:
        idx = V.nonzero()
        log = np.log
    elif type(V) == torch.Tensor:
        id1, *id2 = V.nonzero().t()
        idx = [id1] + id2
        log = torch.log
    else:
        raise TypeError
    nonzeroV = V[idx]
    Vt = V_tilde[idx]
    return (nonzeroV / Vt - log(nonzeroV / Vt) - 1).sum()
