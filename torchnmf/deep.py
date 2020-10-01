import torch
from torch.nn import Parameter
from typing import Union, Iterable
from collections.abc import Iterable as Iterabc
from .base import Base


class BaseComponent(Base):
    def __init__(self,
                 rank,
                 batch=1,
                 W: Union[Iterable[int], torch.Tensor] = None,
                 H: Union[Iterable[int], torch.Tensor] = None,
                 trainable_W=True,
                 trainable_H=True):
        super().__init__()
        self.rank = rank
        self.batch = batch

        if isinstance(W, Iterabc):
            self.register_parameter('W', Parameter(torch.rand(*W)))
        elif isinstance(W, torch.Tensor):
            self.register_parameter('W', Parameter(torch.empty(*W.size()), requires_grad=trainable_W))
            self.W.data.copy_(W)
        else:
            self.register_parameter('W', None)

        if isinstance(H, Iterabc):
            H_shape = (batch,) + tuple(H) if batch > 1 else H
            self.register_parameter('H', Parameter(torch.rand(*H_shape)))
        elif isinstance(W, torch.Tensor):
            H_shape = H.shape
            if batch > 1 and batch != H_shape[0]:
                 H_shape = (batch,) + H_shape
            self.register_parameter('H', Parameter(torch.empty(*H_shape), requires_grad=trainable_H))
            self.H.data.copy_(H)
        else:
            self.register_parameter('H', None)

    def forward(self, H=None, W=None):
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        return self.reconstruct(H, W)

    def reconstruct(self, H, W):
        raise NotImplementedError

    def get_W_positive(self, WH, beta, H_sum) -> (torch.Tensor, None or torch.Tensor):
        raise NotImplementedError

    def get_H_positive(self, WH, beta, W_sum) -> (torch.Tensor, None or torch.Tensor):
        raise NotImplementedError

    @staticmethod
    def get_W_norm(W):
        W2 = W * W
        sum_dims = list(range(W2.dim()))
        sum_dims.remove(1)
        return W2.sum(sum_dims).sqrt()

    @staticmethod
    def get_H_norm(H, batched=False):
        H2 = H * H
        if batched:
            sum_dims = list(range(H2.dim()))
            sum_dims.remove(1)
        else:
            sum_dims = list(range(1, H2.dim()))
        return H2.sum(sum_dims).sqrt()

    @staticmethod
    @torch.no_grad()
    def renorm(W, H, unit_norm='W', batched=False):
        if unit_norm == 'W':
            W_norm = BaseComponent.get_W_norm(W)
            slicer = (slice(None),) + (None,) * (W.dim() - 2)
            W /= W_norm[slicer]
            slicer = (slice(None),) + (None,) * (H.dim() - (2 if batched else 1))
            H *= W_norm[slicer]
        elif unit_norm == 'H':
            H_norm = BaseComponent.get_H_norm(H, batched)
            slicer = (slice(None),) + (None,) * (H.dim() - (2 if batched else 1))
            H /= H_norm[slicer]
            slicer = (slice(None),) + (None,) * (W.dim() - 2)
            W *= H_norm[slicer]
        else:
            raise ValueError("Input type isn't valid!")
