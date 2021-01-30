import torch
from torch.optim.optimizer import Optimizer
from .nmf import _proj_func, _get_norm

eps = 1e-8


class BetaMu(Optimizer):
    r"""Implements the classic multiplicative updater for NMF models minimizing β-divergence.

    Note:
        To use this optimizer, not only make sure your model parameters are non-negative, but the gradients
        along the whole computational graph are always non-negative.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        beta (float, optional): beta divergence to be minimized, measuring the distance between target and the NMF model.
                        Default: ``1.``.
        l1_reg (float, optional): L1 regularize penalty. Default: ``0.``.
        l2_reg (float, optional): L2 regularize penalty (weight decay). Default: ``0.``.
        orthogonal (float, optional): Orthogonal regularize penalty. Default: ``0.``.
    """

    def __init__(self, params, beta=1, l1_reg=0, l2_reg=0, orthogonal=0):
        if not 0.0 <= l1_reg:
            raise ValueError("Invalid l1_reg value: {}".format(l1_reg))
        if not 0.0 <= l2_reg:
            raise ValueError("Invalid l2_reg value: {}".format(l2_reg))
        if not 0.0 <= orthogonal:
            raise ValueError("Invalid orthogonal value: {}".format(orthogonal))
        defaults = dict(beta=beta, l1_reg=l1_reg, l2_reg=l2_reg, orthogonal=orthogonal)
        super(BetaMu, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):
        """Performs a single update step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the target and predicted Tensor in the form:
                ``func()->Tuple(target,predict)``.
        """

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        status_cache = dict()

        for group in self.param_groups:
            for p in group['params']:
                status_cache[id(p)] = p.requires_grad
                p.requires_grad = False

        for group in self.param_groups:
            beta = group['beta']
            l1_reg = group['l1_reg']
            l2_reg = group['l2_reg']
            ortho = group['orthogonal']

            if beta < 1:
                gamma = 1 / (2 - beta)
            elif beta > 2:
                gamma = 1 / (beta - 1)
            else:
                gamma = 1

            for p in group['params']:
                if not status_cache[id(p)]:
                    continue
                p.requires_grad = True

                V, WH = closure()
                if not WH.requires_grad:
                    p.requires_grad = False
                    continue

                if beta == 2:
                    output_neg = V
                    output_pos = WH
                elif beta == 1:
                    output_neg = V / WH
                    output_pos = torch.ones_like(WH)
                elif beta == 0:
                    output_neg = V / (WH * WH)
                    output_pos = 1 / WH
                else:
                    output_neg = WH.pow(beta - 2) * V
                    output_pos = WH.pow(beta - 1)
                # first backward
                WH.backward(output_neg, retain_graph=True)
                neg = torch.clone(p.grad).relu_()
                p.grad.zero_()
                WH.backward(output_pos)
                pos = torch.clone(p.grad).relu_()
                p.grad.add_(-neg)

                if l1_reg > 0:
                    pos.add_(l1_reg)
                if l2_reg > 0:
                    pos.add_(p, alpha=l2_reg)

                if ortho > 0:
                    pos.add_(p.sum(1, keepdims=True) - p, alpha=ortho)

                pos.add_(eps)
                neg.add_(eps)
                multiplier = neg.div_(pos)
                if gamma != 1:
                    multiplier.pow_(gamma)

                p.mul_(multiplier)
                p.requires_grad = False

        for group in self.param_groups:
            for p in group['params']:
                p.requires_grad = status_cache[id(p)]

        return None


class SparsityProj(Optimizer):
    r"""Implements parseness constrainted gradient projection method described in `Non-negative Matrix Factorization
    with Sparseness Constraints`_.

    .. _`Non-negative Matrix Factorization with Sparseness Constraints`:
            https://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf


    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        sparsity (float): the target sparseness for `params`, with 0 < sparsity < 1.
        dim (int, optional): dimension over which to compute the sparseness for each parameter. Default: ``1``.
        max_iter (int, optional): maximal number of function evaluations per optimization step. Default: ``10``.
    """

    def __init__(self, params, sparsity, dim=1, max_iter=10):
        if not 0.0 < sparsity < 1.:
            raise ValueError("Invalid sparsity value: {}".format(sparsity))
        defaults = dict(sparsity=sparsity, lr=1, dim=dim, max_iter=max_iter)
        super(SparsityProj, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):
        """Performs a single update step.

        Arguments:
            closure (callable): A closure that reevaluates the model and returns the loss.
        """
        loss = None

        for group in self.param_groups:
            sparsity = group['sparsity']
            lr = group['lr']
            dim = group['dim']
            max_iter = group['max_iter']

            with torch.enable_grad():
                init_loss = closure()
                init_loss.backward()

            params = [(p, p.grad.clone()) for p in group['params'] if p.grad is not None]

            for i in range(max_iter):
                for p, g in params:
                    norms = _get_norm(p, dim)
                    p.add_(g, alpha=-lr)
                    dim = p.numel() // p.shape[dim]
                    L1 = dim ** 0.5 * (1 - sparsity) + sparsity
                    for j in range(p.shape[dim]):
                        slicer = (slice(None),) * dim + (j,)
                        p[slicer] = _proj_func(p[slicer], L1 * norms[j], norms[j] ** 2)

                loss = closure()
                if loss <= init_loss:
                    break

                for p, g in params:
                    p.add_(g, alpha=lr)
                lr *= 0.5

            lr *= 1.2

            group['lr'] = lr
        return loss
