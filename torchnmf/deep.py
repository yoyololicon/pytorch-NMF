import torch
from torch.nn import Parameter
import torch.nn.functional as F
from typing import Union, Iterable
from collections.abc import Iterable as Iterabc
from .base import Base


class BaseComponent(Base):
    def __init__(self,
                 rank=None,
                 batch=1,
                 W: Union[Iterable[int], torch.Tensor] = None,
                 H: Union[Iterable[int], torch.Tensor] = None,
                 trainable_W=True,
                 trainable_H=True):
        super().__init__()

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

        if isinstance(self.W, torch.Tensor):
            if isinstance(self.H, torch.Tensor):
                assert self.W.shape[1] == self.H.shape[1 if batch > 1 else 0], "Latent size of W and H should be equal!"
            rank = self.W.shape[1]
        elif isinstance(self.H, torch.Tensor):
            rank = self.H.shape[1 if batch > 1 else 0]
        else:
            assert rank, "A rank should be given when both W and H are not available!"

        self.rank = rank
        self.batch = batch

    def forward(self, H=None, W=None):
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        return self.reconstruct(H, W)

    @staticmethod
    def reconstruct(H, W):
        raise NotImplementedError

    @staticmethod
    def get_W_positive(H, WH, beta, *args) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def get_H_positive(W, WH, beta, *args) -> torch.Tensor:
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


class NMF(BaseComponent):
    def __init__(self,
                 Vshape: Iterable[int] = None,
                 **kwargs):
        if isinstance(Vshape, Iterabc):
            K, M = Vshape
            rank = kwargs.get('rank', K)

            inspect_W = (K, rank)
            inspect_H = (rank, M)
            kwargs['W'] = kwargs.get('W', inspect_W)
            kwargs['H'] = kwargs.get('H', inspect_H)
            kwargs['rank'] = rank

        super().__init__(**kwargs)

    @staticmethod
    def reconstruct(H, W):
        return W @ H

    @staticmethod
    def get_W_positive(H, WH, beta):
        if beta == 1:
            H_sum = H.sum(1)
            denominator = H_sum[None, :]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WHHt = WH @ H.t()
            denominator = WHHt

        return denominator

    @staticmethod
    def get_H_positive(W, WH, beta):
        if beta == 1:
            W_sum = W.sum(0)
            denominator = W_sum[:, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WtWH = W.t() @ WH
            denominator = WtWH
        return denominator


class NMFD(BaseComponent):
    def __init__(self,
                 Vshape: Iterable[int] = None,
                 T=1,
                 **kwargs):
        if isinstance(Vshape, Iterabc):
            if len(Vshape) == 3:
                batch, K, M = Vshape
            else:
                K, M = Vshape
                batch = 1
            kwargs['batch'] = batch
            rank = kwargs.get('rank', K)

            inspect_W = (K, rank, T)
            inspect_H = (rank, M - T + 1)
            kwargs['W'] = kwargs.get('W', inspect_W)
            kwargs['H'] = kwargs.get('H', inspect_H)
            kwargs['rank'] = rank

        super().__init__(**kwargs)

    @staticmethod
    def reconstruct(H, W):
        if H.dim() < 3:
            H = H.unsqueeze(0)
        pad_size = W.shape[2] - 1
        return F.conv1d(H, W.flip(2), padding=pad_size).squeeze(0)

    @staticmethod
    def get_W_positive(H, WH, beta, *args) -> torch.Tensor:
        if H.dim() < 3:
            H = H.unsqueeze(0)
        batch, rank, _ = H.shape
        *_, K, M = WH.shape

        if beta == 1:
            denominator = H.sum((0, 2), keepdims=True)
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            if batch > 1:
                WH = WH.view(1, batch * K, M)
                H = H.unsqueeze(1).expand(-1, K, -1, -1).view(-1, 1, H.shape[-1])
                WHHt = F.conv1d(WH, H, groups=batch * K)
                WHHt = WHHt.view(batch, K, rank, WHHt.shape[-1]).sum(0)
            else:
                WHHt = F.conv1d(WH.view(K, 1, M), H.transpose(0, 1))
            denominator = WHHt

        return denominator

    @staticmethod
    def get_H_positive(W, WH, beta, *args) -> torch.Tensor:
        if beta == 1:
            W_sum = W.sum((0, 2))
            denominator = W_sum[:, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            if len(WH.shape) < 3:
                WH = WH.unsqueeze(0)
            WtWH = F.conv1d(WH, W.transpose(0, 1)).squeeze(0)
            denominator = WtWH
        return denominator


class NMF2D(BaseComponent):
    def __init__(self,
                 Vshape: Iterable[int] = None,
                 kernel: Union[Iterable[int], int] = 1,
                 **kwargs):
        if isinstance(Vshape, Iterabc):
            try:
                F, T = kernel
            except:
                F = T = kernel
            if len(Vshape) == 4:
                batch, channel, K, M = Vshape
            else:
                channel, K, M = Vshape
                batch = 1
            kwargs['batch'] = batch
            rank = kwargs.get('rank', K)

            inspect_W = (channel, rank, F, T)
            inspect_H = (rank, K - F + 1, M - T + 1)
            kwargs['W'] = kwargs.get('W', inspect_W)
            kwargs['H'] = kwargs.get('H', inspect_H)
            kwargs['rank'] = rank

        super().__init__(**kwargs)

    @staticmethod
    def reconstruct(H, W):
        if H.dim() < 4:
            H = H.unsqueeze(0)
        pad_size = (W.shape[2] - 1, W.shape[3] - 1)
        out = F.conv2d(H, W.flip((2, 3)), padding=pad_size).squeeze(0)
        return out

    @staticmethod
    def get_W_positive(H, WH, beta, *args) -> torch.Tensor:
        if H.dim() < 4:
            H = H.unsqueeze(0)
        batch, rank, _ = H.shape
        *_, channel, K, M = WH.shape

        if beta == 1:
            denominator = H.sum((0, 2, 3), keepdims=True)
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            if batch > 1:
                WH = WH.view(1, batch * channel, K, M)
                # batch * channel => batch * channel * rank
                # H.shape = (batch * channel * rank, 1
                H = H.unsqueeze(1).expand(-1, channel, -1, -1, -1).view(-1, 1, *H.shape[-2:])
                WHHt = F.conv2d(WH, H, groups=batch * channel)
                WHHt = WHHt.view(batch, channel, rank, *WHHt.shape[-2:]).sum(0)
            else:
                WH = WH.view(channel, 1, K, M)
                WHHt = F.conv2d(WH, H.transpose(0, 1))
            denominator = WHHt

        return denominator

    @staticmethod
    def get_H_positive(W, WH, beta, *args) -> torch.Tensor:
        if beta == 1:
            W_sum = W.sum((0, 2, 3))
            denominator = W_sum[:, None, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            if len(WH.shape) < 4:
                WH = WH.unsqueeze(0)
            WtWH = F.conv2d(WH, W.transpose(0, 1)).squeeze(0)
            denominator = WtWH
        return denominator


class NMF3D(BaseComponent):
    def __init__(self,
                 Vshape: Iterable[int] = None,
                 kernel: Union[Iterable[int], int] = 1,
                 **kwargs):
        if isinstance(Vshape, Iterabc):
            try:
                T, H, W = kernel
            except:
                T = H = W = kernel
            if len(Vshape) == 5:
                batch, channel, N, K, M = Vshape
            else:
                channel, N, K, M = Vshape
                batch = 1
            kwargs['batch'] = batch
            rank = kwargs.get('rank', K)

            inspect_W = (channel, rank, T, H, W)
            inspect_H = (rank, N - T + 1, K - H + 1, M - W + 1)
            kwargs['W'] = kwargs.get('W', inspect_W)
            kwargs['H'] = kwargs.get('H', inspect_H)
            kwargs['rank'] = rank

        super().__init__(**kwargs)

    @staticmethod
    def reconstruct(H, W):
        if H.dim() < 5:
            H = H.unsqueeze(0)
        pad_size = (W.shape[2] - 1, W.shape[3] - 1, W.shape[4] - 1)
        out = F.conv3d(H, W.flip((2, 3, 4)), padding=pad_size).squeeze(0)
        return out

    @staticmethod
    def get_W_positive(H, WH, beta, *args) -> torch.Tensor:
        if H.dim() < 5:
            H = H.unsqueeze(0)
        batch, rank, _ = H.shape
        *_, channel, N, K, M = WH.shape

        if beta == 1:
            denominator = H.sum((0, 2, 3, 4), keepdims=True)
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            if batch > 1:
                WH = WH.view(1, batch * channel, N, K, M)
                H = H.unsqueeze(1).expand(-1, channel, -1, -1, -1, -1).view(-1, 1, *H.shape[-3:])
                WHHt = F.conv3d(WH, H, groups=batch * channel)
                WHHt = WHHt.view(batch, channel, rank, *WHHt.shape[-3:]).sum(0)
            else:
                WH = WH.view(channel, 1, N, K, M)
                WHHt = F.conv3d(WH, H.transpose(0, 1))
            denominator = WHHt

        return denominator

    @staticmethod
    def get_H_positive(W, WH, beta, *args) -> torch.Tensor:
        if beta == 1:
            W_sum = W.sum((0, 2, 3, 4))
            denominator = W_sum[:, None, None, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            if len(WH.shape) < 5:
                WH = WH.unsqueeze(0)
            WtWH = F.conv3d(WH, W.transpose(0, 1)).squeeze(0)
            denominator = WtWH
        return denominator
