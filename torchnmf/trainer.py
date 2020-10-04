import torch
from torch import nn
import torch.nn.functional as F
from .deep import BaseComponent
from .metrics import Beta_divergence
from collections import defaultdict

eps = 1e-8


def _store_in_out(module, input, output) -> None:
    W, H = input
    if W is None:
        W = getattr(module, 'W')
    if H is None:
        H = getattr(module, 'H')

    if H.requires_grad or W.requires_grad or output.requires_grad:
        setattr(module, '_W_cache', W)
        setattr(module, '_H_cache', H)
        setattr(module, '_WH_cache', output)
    else:
        if hasattr(module, '_W_cache'):
            delattr(module, '_W_cache')
        if hasattr(module, '_H_cache'):
            delattr(module, '_H_cache')
        if hasattr(module, '_WH_cache'):
            delattr(module, '_WH_cache')


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

        losses = []
        for p in self.params:
            p.requires_grad = True

            # forward
            WH = self.model(*inputs, **kwargs)#.clamp_min(eps)
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
        return Beta_divergence(WH, V + eps, self.beta).item() /  V.numel()