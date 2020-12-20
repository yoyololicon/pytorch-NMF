import torch
from torch.nn import functional as F


def KL_divergence(predict, target):
    return F.kl_div(predict.add(1e-8).log(), target, reduction='sum') - target.sum() + predict.sum()

def Euclidean(predict, target):
    return F.mse_loss(predict, target, reduction='sum') * 0.5


def IS_divergence(predict, target):
    div = target / predict.add(1e-8)
    return div.sum() - div.add(1e-8).log().sum() - target.numel()


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


def sparseness(x):
    N = x.numel()
    return (N ** 0.5 - x.norm(1) / x.norm(2)) / (N ** 0.5 - 1)


if __name__ == '__main__':
    x = torch.rand(5, 5)
    y = torch.rand_like(x)
    print((y * (y / x).log()).sum(), F.kl_div(x.log(), y, reduction='sum'))
    x.requires_grad = True

    loss = Euclidean(x, y)
    loss.backward()
    print(x.grad, loss.item())
    x.grad.zero_()

    loss = Beta_divergence(x, y)
    loss.backward()
    print(x.grad, loss.item())