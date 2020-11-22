import torch
from torch.nn import Parameter
import torch.nn.functional as F
from typing import Union, Iterable, Optional, List, Tuple
from collections.abc import Iterable as Iterabc
from .base import Base
from .metrics import Beta_divergence
from tqdm import tqdm


def _proj_func(s: torch.Tensor, k1: float, k2: float):
    s_shape = s.shape
    s = s.contiguous().view(-1)
    N = s.numel()
    v = s + (k1 - s.sum()) / N

    zero_coef = torch.ones_like(v) < 0
    while True:
        m = k1 / (N - zero_coef.sum())
        w = torch.where(~zero_coef, v - m, v)
        a = w @ w
        b = 2 * w @ v
        c = v @ v - k2
        alphap = (-b + torch.clamp(b * b - 4 * a * c, min=0).sqrt()) * 0.5 / a
        v += alphap * w

        if (v >= 0).all():
            break

        zero_coef |= v < 0
        v[zero_coef] = 0
        v += (k1 - v.sum()) / (N - zero_coef.sum())
        v[zero_coef] = 0

    return v.view(*s_shape)


def _double_backward_update(V, WH, param, beta, gamma, l1_reg, l2_reg, pos=None):
    param.grad = None
    if beta == 2:
        output_neg = V
        output_pos = WH
    elif beta == 1:
        output_neg = V / WH
        output_pos = None
    else:
        output_neg = WH.pow(beta - 2) * V
        output_pos = WH.pow(beta - 1)
    # first backward
    WH.backward(output_neg, retain_graph=pos is None)
    neg = torch.clone(param.grad).detach()

    if pos is None:
        param.grad.zero_()
        WH.backward(output_pos)
        pos = torch.clone(param.grad).detach()

    if l1_reg > 0:
        pos += l1_reg
    if l2_reg > 0:
        pos += param.data * l2_reg
    multiplier = neg / (pos + 1e-8)
    if gamma != 1:
        multiplier.pow_(gamma)
    param.data.mul_(multiplier)


class BaseComponent(Base):
    __constants__ = ['rank']
    __annotations__ = {'W': Optional[torch.Tensor],
                       'H': Optional[torch.Tensor],
                       'out_channels': Optional[int],
                       'kernel_size': Optional[Tuple[int, ...]]}

    rank: int
    W: Optional[torch.Tensor]
    H: Optional[torch.Tensor]
    out_channels: Optional[int]
    kernel_size: Optional[Tuple[int, ...]]

    def __init__(self,
                 rank=None,
                 W: Union[Iterable[int], torch.Tensor] = None,
                 H: Union[Iterable[int], torch.Tensor] = None,
                 trainable_W=True,
                 trainable_H=True):
        super().__init__()

        if isinstance(W, torch.Tensor):
            self.register_parameter('W', Parameter(torch.empty(*W.size()), requires_grad=trainable_W))
            self.W.data.copy_(W)
        elif isinstance(W, Iterabc) and trainable_W:
            self.register_parameter('W', Parameter(torch.randn(*W).abs()))
        else:
            self.register_parameter('W', None)

        if isinstance(H, torch.Tensor):
            H_shape = H.shape
            self.register_parameter('H', Parameter(torch.empty(*H_shape), requires_grad=trainable_H))
            self.H.data.copy_(H)
        elif isinstance(H, Iterabc) and trainable_H:
            self.register_parameter('H', Parameter(torch.randn(*H).abs()))
        else:
            self.register_parameter('H', None)

        if isinstance(self.W, torch.Tensor):
            if isinstance(self.H, torch.Tensor):
                assert self.W.shape[1] == self.H.shape[1], "Latent size of W and H should be equal!"
            rank = self.W.shape[1]
            self.out_channels = self.W.shape[0]
            if len(self.W.shape) > 2:
                self.kernel_size = tuple(self.W.shape[2:])
        elif isinstance(self.H, torch.Tensor):
            rank = self.H.shape[1]
        else:
            assert rank, "A rank should be given when both W and H are not available!"

        self.rank = rank

    def extra_repr(self) -> str:
        s = ('{rank}')
        if self.W is not None:
            s += ', out_channels={out_channels}'
            if hasattr(self, 'kernel_size'):
                s += ', kernel_size={kernel_size}'
        return s.format(**self.__dict__)

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
    def get_W_kl_positive(H) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def get_H_kl_positive(W) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def get_W_norm(W) -> torch.Tensor:
        W2 = W * W
        sum_dims = list(range(W2.dim()))
        sum_dims.remove(1)
        return W2.sum(sum_dims).sqrt()

    @staticmethod
    def get_H_norm(H) -> torch.Tensor:
        H2 = H * H
        sum_dims = list(range(H2.dim()))
        sum_dims.remove(1)
        return H2.sum(sum_dims).sqrt()

    @staticmethod
    @torch.no_grad()
    def renorm(W, H, unit_norm='W'):
        if unit_norm == 'W':
            W_norm = BaseComponent.get_W_norm(W)
            slicer = (slice(None),) + (None,) * (W.dim() - 2)
            W /= W_norm[slicer]
            slicer = (slice(None),) + (None,) * (H.dim() - 2)
            H *= W_norm[slicer]
        elif unit_norm == 'H':
            H_norm = BaseComponent.get_H_norm(H)
            slicer = (slice(None),) + (None,) * (H.dim() - 2)
            H /= H_norm[slicer]
            slicer = (slice(None),) + (None,) * (W.dim() - 2)
            W *= H_norm[slicer]
        else:
            raise ValueError("Input type isn't valid!")

    def fit(self,
            V,
            W=None,
            H=None,
            beta=1,
            tol=1e-4,
            max_iter=200,
            verbose=0,
            alpha=0,
            l1_ratio=0
            ):
        if W is None:
            W = self.W
        if H is None:
            H = self.H

        if beta < 1:
            gamma = 1 / (2 - beta)
        elif beta > 2:
            gamma = 1 / (beta - 1)
        else:
            gamma = 1

        l1_reg = alpha * l1_ratio
        l2_reg = alpha * (1 - l1_ratio)

        with torch.no_grad():
            WH = self.reconstruct(H, W)
            loss_init = previous_loss = Beta_divergence(WH, V, beta).mul(2).sqrt().item()

        with tqdm(total=max_iter, disable=not verbose) as pbar:
            for n_iter in range(max_iter):
                if W.requires_grad:
                    WH = self.reconstruct(H.detach(), W)
                    _double_backward_update(V, WH, W, beta, gamma, l1_reg, l2_reg,
                                            self.get_W_kl_positive(H.detach()) if beta == 1 else None)

                if H.requires_grad:
                    WH = self.reconstruct(H, W.detach())
                    _double_backward_update(V, WH, H, beta, gamma, l1_reg, l2_reg,
                                            self.get_H_kl_positive(W.detach()) if beta == 1 else None)

                if n_iter % 10 == 9:
                    with torch.no_grad():
                        WH = self.reconstruct(H, W)
                        loss = Beta_divergence(WH, V, beta).mul(2).sqrt().item()
                    pbar.set_postfix(loss=loss)
                    # pbar.set_description('Beta loss=%.4f' % error)
                    pbar.update(10)
                    if (previous_loss - loss) / loss_init < tol:
                        break
                    previous_loss = loss

        return n_iter

    def sparse_fit(self,
                   V,
                   W=None,
                   H=None,
                   beta=1,
                   max_iter=200,
                   verbose=0,
                   sW=None,
                   sH=None,
                   ):
        if W is None:
            W = self.W
        if H is None:
            H = self.H

        if sW is not None and W.requires_grad:
            dim = W[:, 0].numel()
            L1a = dim ** 0.5 * (1 - sW) + sW
            with torch.no_grad():
                for i in range(W.shape[1]):
                    W[:, i] = _proj_func(W[:, i], L1a, 1)
        else:
            L1a = None

        if sH is not None and H.requires_grad:
            dim = H[:, 0].numel()
            L1s = dim ** 0.5 * (1 - sH) + sH
            with torch.no_grad():
                for j in range(H.shape[1]):
                    H[:, j] = _proj_func(H[:, j], L1s, 1)
        else:
            L1s = None

        if beta < 1:
            gamma = 1 / (2 - beta)
        elif beta > 2:
            gamma = 1 / (beta - 1)
        else:
            gamma = 1

        stepsize_W, stepsize_H = 1, 1
        with tqdm(total=max_iter, disable=not verbose) as pbar:
            for n_iter in range(max_iter):
                if W.requires_grad:
                    if sW is None:
                        WH = self.reconstruct(H.detach(), W)
                        _double_backward_update(V, WH, W, beta, gamma, 0, 0,
                                                self.get_W_kl_positive(H.detach()) if beta == 1 else None)
                    else:
                        W.grad = None
                        WH = self.reconstruct(H.detach(), W)
                        loss = Beta_divergence(self.fix_neg(WH), V, beta)
                        loss.backward()
                        with torch.no_grad():
                            for i in range(10):
                                Wnew = W - stepsize_W * W.grad
                                norms = BaseComponent.get_W_norm(Wnew)
                                for j in range(Wnew.shape[1]):
                                    Wnew[:, j] = _proj_func(Wnew[:, j], L1a * norms[j], norms[j] ** 2)
                                new_loss = Beta_divergence(self.fix_neg(self.reconstruct(self.H, Wnew)),
                                                           V, beta)
                                if new_loss <= loss:
                                    break

                                stepsize_W *= 0.5

                            stepsize_W *= 1.2
                            W.copy_(Wnew)

                if H.requires_grad:
                    if sH is None:
                        WH = self.reconstruct(H, W.detach())
                        _double_backward_update(V, WH, H, beta, gamma, 0, 0,
                                                self.get_H_kl_positive(W.detach()) if beta == 1 else None)
                    else:
                        H.grad = None
                        WH = self.reconstruct(H, W.detach())
                        loss = Beta_divergence(self.fix_neg(WH), V, beta)
                        loss.backward()

                        with torch.no_grad():
                            for i in range(10):
                                Hnew = H - stepsize_H * H.grad
                                norms = BaseComponent.get_H_norm(Hnew)
                                for j in range(H.shape[1]):
                                    Hnew[:, j] = _proj_func(Hnew[:, j], L1s * norms[j], norms[j] ** 2)
                                new_loss = Beta_divergence(self.fix_neg(self.reconstruct(Hnew, W)),
                                                           V, beta)
                                if new_loss <= loss:
                                    break

                                stepsize_H *= 0.5

                            stepsize_H *= 1.2
                            H.copy_(Hnew)

                        BaseComponent.renorm(W, H, 'W')

                if n_iter % 10 == 9:
                    with torch.no_grad():
                        WH = self.reconstruct(H, W)
                        loss = Beta_divergence(WH, V, beta).mul(2).sqrt().item()
                    pbar.set_postfix(loss=loss)
                    pbar.update(10)
        return n_iter


class NMF(BaseComponent):
    def __init__(self,
                 Vshape: Iterable[int] = None,
                 **kwargs):
        if isinstance(Vshape, Iterabc):
            M, K = Vshape
            if 'W' in kwargs and 'H' not in kwargs:
                if isinstance(kwargs['W'], torch.Tensor):
                    rank = kwargs['W'].shape[1]
                elif isinstance(kwargs['W'], Iterabc):
                    rank = kwargs['W'][1]
                else:
                    rank = None
                if rank:
                    kwargs['H'] = (M, rank)
            elif 'H' in kwargs and 'W' not in kwargs:
                if isinstance(kwargs['H'], torch.Tensor):
                    rank = kwargs['H'].shape[1]
                elif isinstance(kwargs['H'], Iterabc):
                    rank = kwargs['H'][1]
                else:
                    rank = None
                if rank:
                    kwargs['W'] = (K, rank)
            else:
                kwargs['rank'] = kwargs.get('rank', K)

        super().__init__(**kwargs)

    @staticmethod
    def reconstruct(H, W):
        return F.linear(H, W)

    @staticmethod
    def get_W_kl_positive(H):
        return H.sum(0, keepdim=True)

    @staticmethod
    def get_H_kl_positive(W):
        return W.sum(0, keepdim=True)


class NMFD(BaseComponent):
    def __init__(self,
                 Vshape: Iterable[int] = None,
                 T=1,
                 **kwargs):
        if isinstance(Vshape, Iterabc):
            batch, K, M = Vshape
            if 'W' in kwargs and 'H' not in kwargs:
                if isinstance(kwargs['W'], torch.Tensor):
                    rank, T = kwargs['W'].shape[1:]
                elif isinstance(kwargs['W'], Iterabc):
                    rank, T = kwargs['W'][1:]
                else:
                    rank = None
                if rank:
                    kwargs['H'] = (batch, rank, M - T + 1)
            elif 'H' in kwargs and 'W' not in kwargs:
                if isinstance(kwargs['H'], torch.Tensor):
                    rank, M_T = kwargs['H'].shape[1:]
                elif isinstance(kwargs['H'], Iterabc):
                    rank, M_T = kwargs['H'][1:]
                else:
                    rank = None
                if rank:
                    kwargs['W'] = (K, rank, M - M_T + 1)
            else:
                kwargs['rank'] = kwargs.get('rank', K)

        super().__init__(**kwargs)

    @staticmethod
    def reconstruct(H, W):
        pad_size = W.shape[2] - 1
        return F.conv1d(H, W.flip(2), padding=pad_size)

    @staticmethod
    def get_W_kl_positive(H) -> torch.Tensor:
        return H.sum((0, 2), keepdims=True)

    @staticmethod
    def get_H_kl_positive(W) -> torch.Tensor:
        W_sum = W.sum((0, 2))
        return W_sum[:, None]


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
            batch, channel, K, M = Vshape

            if 'W' in kwargs and 'H' not in kwargs:
                if isinstance(kwargs['W'], torch.Tensor):
                    rank, F, T = kwargs['W'].shape[1:]
                elif isinstance(kwargs['W'], Iterabc):
                    rank, F, T = kwargs['W'][1:]
                else:
                    rank = None
                if rank:
                    kwargs['H'] = (batch, rank, K - F + 1, M - T + 1)
            elif 'H' in kwargs and 'W' not in kwargs:
                if isinstance(kwargs['H'], torch.Tensor):
                    rank, K_F, M_T = kwargs['H'].shape[1:]
                elif isinstance(kwargs['H'], Iterabc):
                    rank, K_F, M_T = kwargs['H'][1:]
                else:
                    rank = None
                if rank:
                    kwargs['W'] = (channel, rank, K - K_F + 1, M - M_T + 1)
            else:
                kwargs['rank'] = kwargs.get('rank', K)

        super().__init__(**kwargs)

    @staticmethod
    def reconstruct(H, W):
        pad_size = (W.shape[2] - 1, W.shape[3] - 1)
        out = F.conv2d(H, W.flip((2, 3)), padding=pad_size)
        return out

    @staticmethod
    def get_W_kl_positive(H) -> torch.Tensor:
        return H.sum((0, 2, 3), keepdims=True)

    @staticmethod
    def get_H_kl_positive(W) -> torch.Tensor:
        W_sum = W.sum((0, 2, 3))
        return W_sum[:, None, None]


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
            batch, channel, N, K, M = Vshape

            if 'W' in kwargs and 'H' not in kwargs:
                if isinstance(kwargs['W'], torch.Tensor):
                    rank, N, K, M = kwargs['W'].shape[1:]
                elif isinstance(kwargs['W'], Iterabc):
                    rank, N, K, M = kwargs['W'][1:]
                else:
                    rank = None
                if rank:
                    kwargs['H'] = (batch, rank, N - T + 1, K - H + 1, M - W + 1)
            elif 'H' in kwargs and 'W' not in kwargs:
                if isinstance(kwargs['H'], torch.Tensor):
                    rank, N_T, K_H, M_W = kwargs['H'].shape[1:]
                elif isinstance(kwargs['H'], Iterabc):
                    rank, N_T, K_H, M_W = kwargs['H'][1:]
                else:
                    rank = None
                if rank:
                    kwargs['W'] = (channel, rank, N - N_T + 1, K - K_H + 1, M - M_W + 1)
            else:
                kwargs['rank'] = kwargs.get('rank', K)

        super().__init__(**kwargs)

    @staticmethod
    def reconstruct(H, W):
        pad_size = (W.shape[2] - 1, W.shape[3] - 1, W.shape[4] - 1)
        out = F.conv3d(H, W.flip((2, 3, 4)), padding=pad_size)
        return out

    @staticmethod
    def get_W_kl_positive(H) -> torch.Tensor:
        return H.sum((0, 2, 3, 4), keepdims=True)

    @staticmethod
    def get_H_kl_positive(W) -> torch.Tensor:
        W_sum = W.sum((0, 2, 3, 4))
        return W_sum[:, None, None, None]
