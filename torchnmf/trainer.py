import torch
from torch import nn
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
from .deep import BaseComponent
from .metrics import Beta_divergence
from collections import defaultdict

eps = 1e-8


class BetaMu(Optimizer):
    def __init__(self, params, beta=1, l1_reg=0, l2_reg=0, sparsity=None):
        if not 0.0 <= l1_reg:
            raise ValueError("Invalid l1_reg value: {}".format(l1_reg))
        if not 0.0 <= l2_reg:
            raise ValueError("Invalid l2_reg value: {}".format(l2_reg))
        if sparsity is not None and not 0.0 < sparsity < 1.:
            raise ValueError("Invalid sparsity value: {}".format(sparsity))
        defaults = dict(beta=beta, l1_reg=l1_reg, l2_reg=l2_reg, sparsity=sparsity)
        super(BetaMu, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the target and predict Tensor.
        """

        # Make sure the closure is always called with grad enabled
        closure = torch.enable_grad()(closure)

        status_cache = dict()

        orig_loss = None

        for group in self.param_groups:
            for p in group['params']:
                status_cache[id(p)] = p.requires_grad
                p.requires_grad = False

        for group in self.param_groups:
            beta = group['beta']
            l1_reg = group['l1_reg']
            l2_reg = group['l2_reg']
            sparsity = group['sparsity']

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
                elif beta == -1:
                    output_neg = V / (WH * WH)
                    output_pos = 1 / WH
                else:
                    output_neg = WH.pow(beta - 2) * V
                    output_pos = WH.pow(beta - 1)
                # first backward
                WH.backward(output_neg, retain_graph=True)
                neg = torch.clone(p.grad).detach()
                p.grad.zero_()
                WH.backward(output_pos)
                pos = torch.clone(p.grad).detach()
                p.grad.add_(-neg)

                if l1_reg > 0:
                    pos.add_(l1_reg)
                if l2_reg > 0:
                    pos.add_(p, alpha=l2_reg)

                pos.add_(eps)
                multiplier = neg.div_(pos)
                if gamma != 1:
                    multiplier.pow_(gamma)

                p.mul_(multiplier)
                p.requires_grad = False

        for group in self.param_groups:
            for p in group['params']:
                p.requires_grad = status_cache[id(p)]

        return None


class BetaTrainer(object):
    def __init__(self,
                 model: nn.Module,
                 beta: float = 1.,
                 alpha=0,
                 l1_ratio=0):
        nmf_modules = [m for m in model.modules() if isinstance(m, BaseComponent)]
        assert len(nmf_modules), "Model should have a least one NMF submodule"

        params = [p for m in nmf_modules for p in m.parameters() if p.requires_grad and p.is_leaf]
        assert len(params), "Model should have a least one NMF parameter that requires gradients"

        self.model = model
        self.modules = nmf_modules
        self.params = params
        self.state = defaultdict(dict)

        self.beta = beta

        if beta < 1:
            gamma = 1 / (2 - beta)
        elif beta > 2:
            gamma = 1 / (beta - 1)
        else:
            gamma = 1
        self.gamma = gamma
        self.l1_reg = alpha * l1_ratio
        self.l2_reg = alpha * (1 - l1_ratio)

    def step(self,
             V: torch.Tensor,
             *inputs, **kwargs):

        for p in self.params:
            p.requires_grad = False

        for p in self.params:
            p.requires_grad = True

            # forward
            WH = self.model(*inputs, **kwargs)  # .clamp_min(eps)
            if not WH.requires_grad:
                p.requires_grad = False
                continue

            if self.beta == 2:
                output_neg = V
                output_pos = WH
            elif self.beta == 1:
                output_neg = V / WH
                output_pos = torch.ones_like(WH)
            else:
                output_neg = WH.pow(self.beta - 2) * V
                output_pos = WH.pow(self.beta - 1)
            # first backward
            WH.backward(output_neg, retain_graph=True)
            neg = torch.clone(p.grad).detach()
            p.grad.zero_()

            WH.backward(output_pos)
            pos = torch.clone(p.grad).detach()
            if neg.min() < 0:
                print(neg.min().item())
            neg.clamp_min_(eps)

            if self.l1_reg > 0:
                pos += self.l1_reg
            if self.l2_reg > 0:
                pos += p.data * self.l2_reg
            multiplier = neg / (pos + eps)
            if self.gamma != 1:
                multiplier.pow_(self.gamma)

            p.data.mul_(multiplier)

            p.grad.zero_()
            p.requires_grad = False

        for p in self.params:
            p.requires_grad = True

        WH = self.model(*inputs, **kwargs)
        return Beta_divergence(WH, V + eps, self.beta).item() / V.numel()
