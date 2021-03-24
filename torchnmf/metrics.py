from torch import Tensor
from torch.nn import functional as F
from .constants import eps


def kl_div(input: Tensor, target: Tensor) -> Tensor:
    r"""The generalized `Kullback-Leibler divergence Loss
    <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>`__, which equal to β-divergence loss when β = 1.

    The loss can be described as:

    .. math::
        \ell(x, y) = \sum_{n = 0}^{N - 1} x_n log(\frac{x_n}{y_n}) - x_n + y_n

    Args:
        input (Tensor): tensor of arbitrary shape
        target (Tensor): tensor of the same shape as input

    Returns:
        Tensor: single element tensor
    """
    return target.reshape(-1) @ (target.add(eps).log() - input.add(eps).log()).reshape(-1) - target.sum() + input.sum()


def euclidean(input: Tensor, target: Tensor) -> Tensor:
    r"""The `Euclidean distance
    <https://en.wikipedia.org/wiki/Euclidean_distance>`__, which equal to β-divergence loss when β = 2.

    .. math::
        \ell(x, y) = \frac{1}{2} \sum_{n = 0}^{N - 1} (x_n - y_n)^2

    Args:
        input (Tensor): tensor of arbitrary shape
        target (Tensor): tensor of the same shape as input

    Returns:
        Tensor: single element tensor
    """
    return F.mse_loss(input, target, reduction='sum') * 0.5


def is_div(input: Tensor, target: Tensor) -> Tensor:
    r"""The `Itakura–Saito divergence
    <https://en.wikipedia.org/wiki/Itakura%E2%80%93Saito_distance>`__, which equal to β-divergence loss when β = 0.

    .. math::
        \ell(x, y) = \sum_{n = 0}^{N - 1} \frac{x_n}{y_n} -  log(\frac{x_n}{y_n}) - 1

    Args:
        input (Tensor): tensor of arbitrary shape
        target (Tensor): tensor of the same shape as input

    Returns:
        Tensor: single element tensor
    """
    target_eps, input_eps = target.add(eps), input.add(eps)
    return (target_eps / input_eps).sum() - target_eps.log().sum() + input_eps.log().sum() - target.numel()


def beta_div(input, target, beta=2):
    r"""The `β-divergence loss
    <https://arxiv.org/pdf/1010.1763.pdf>`__ measure

    The loss can be described as:

    .. math::
        \ell(x, y) = \sum_{n = 0}^{N - 1} \frac{1}{\beta (\beta - 1)}\left ( x_n^{\beta} + \left (\beta - 1
        \right ) y_n^{\beta} - \beta x_n y_n^{\beta-1}\right )

    Args:
        input (Tensor): tensor of arbitrary shape
        target (Tensor): tensor of the same shape as input
        beta (float): a real value control the shape of loss function

    Returns:
        Tensor: single element tensor
    """
    if beta == 2:
        return euclidean(input, target)
    elif beta == 1:
        return kl_div(input, target)
    elif beta == 0:
        return is_div(input, target)
    else:
        input = input.reshape(-1).add(eps)
        target = target.reshape(-1)
        if beta < 0:
            target = target.add(eps)
        bminus = beta - 1

        term_1 = target.pow(beta).sum()
        term_2 = input.pow(beta).sum()
        term_3 = target @ input.pow(bminus)

        loss = term_1 + bminus * term_2 - beta * term_3
        return loss / (beta * bminus)


def sparseness(x: Tensor) -> Tensor:
    r"""The sparseness measure proposed in
    `Non-negative Matrix Factorization with Sparseness Constraints
    <https://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf>`__, can be caculated as:

    .. math::
        f(x) = \frac{\sqrt{N} - \frac{\sum_{n=0}^{N-1} |x_n|}{\sqrt{\sum_{n=0}^{N-1} x_n^2}}}{\sqrt{N} - 1}


    Args:
        x (Tensor): tensor of arbitrary shape

    Returns:
        Tensor: single element tensor with value range between 0 (the most sparse) to 1 (the most dense)
    """
    N = x.numel()
    return (N ** 0.5 - x.norm(1) / x.norm(2)) / (N ** 0.5 - 1)
