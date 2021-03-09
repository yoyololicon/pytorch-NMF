import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from .utils import normalize
from .nmf import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple
from tqdm import tqdm
from typing import Union, Iterable, Optional, List, Tuple
from collections.abc import Iterable as Iterabc
from .metrics import kl_div

__all__ = [
    'PLCA', 'SIPLCA', 'SIPLCA2', 'SIPLCA3'
]

eps = 1e-8


def _log_probability(V, WZH, W, Z, H, W_alpha, Z_alpha, H_alpha):
    return V.view(-1) @ WZH.log().view(-1) + W.log().sum().mul(W_alpha - 1) + H.log().sum().mul(
        H_alpha - 1) + Z.log().sum().mul(Z_alpha - 1)


@torch.no_grad()
def get_norm(x: Tensor):
    if x.ndim > 1:
        sum_dims = list(range(x.dim()))
        sum_dims.remove(1)
        norm = x.sum(sum_dims, keepdim=True)
    else:
        norm = x.sum()
    return norm


class BaseComponent(torch.nn.Module):
    r"""Base class for all NMF modules.

    You can't use this module directly.
    Your models should also subclass this class.

    Args:
        rank (int): Size of hidden dimension
        W (tuple or Tensor): Size or initial weights of template tensor W
        H (tuple or Tensor): Size or initial weights of activation tensor H
        trainable_W (bool):  If ``True``, the template tensor W is learnable. Default: ``True``
        trainable_H (bool):  If ``True``, the activation tensor H is learnable. Default: ``True``

    Attributes:
        W (Tensor or None): the template tensor of the module if corresponding argument is given.
            The values are initialized non-negatively.
        H (Tensor or None): the activation tensor of the module if corresponding argument is given.
            The values are initialized non-negatively.

       """
    __constants__ = ['rank']
    __annotations__ = {'W': Optional[Tensor],
                       'H': Optional[Tensor],
                       'Z': Optional[Tensor],
                       'out_channels': Optional[int],
                       'kernel_size': Optional[Tuple[int, ...]]}

    rank: int
    W: Optional[Tensor]
    H: Optional[Tensor]
    Z: Optional[Tensor]
    out_channels: Optional[int]
    kernel_size: Optional[Tuple[int, ...]]

    def __init__(self,
                 rank: int = None,
                 W: Union[Iterable[int], Tensor] = None,
                 H: Union[Iterable[int], Tensor] = None,
                 Z: Tensor = None,
                 trainable_W: bool = True,
                 trainable_H: bool = True,
                 trainable_Z: bool = True):
        super().__init__()

        infer_rank = None
        if isinstance(W, Tensor):
            assert torch.all(W >= 0.), "Tensor W should be non-negative."
            self.register_parameter('W', Parameter(
                torch.empty(*W.size()), requires_grad=trainable_W))
            self.W.data.copy_(W)
        elif isinstance(W, Iterabc):
            self.register_parameter('W', Parameter(torch.randn(*W).abs()))
        else:
            self.register_parameter('W', None)

        if hasattr(self, "W"):
            self.W.data.div_(get_norm(self.W))
            infer_rank = self.W.shape[1]

        if isinstance(H, Tensor):
            assert torch.all(H >= 0.), "Tensor H should be non-negative."
            H_shape = H.shape
            self.register_parameter('H', Parameter(
                torch.empty(*H_shape), requires_grad=trainable_H))
            self.H.data.copy_(H)
        elif isinstance(H, Iterabc):
            self.register_parameter('H', Parameter(torch.randn(*H).abs()))
        else:
            self.register_parameter('H', None)

        if hasattr(self, "H"):
            self.H.data.div_(get_norm(self.H))
            infer_rank = self.H.shape[1]

        if isinstance(Z, Tensor):
            assert Z.ndim == 1, "Z should be one dimensional."
            assert torch.all(Z >= 0.), "Tensor Z should be non-negative."
            rank = Z.numel()
            self.register_parameter('Z', Parameter(
                torch.empty(rank), requires_grad=trainable_Z))
            self.Z.data.copy_(Z)
        elif isinstance(rank, int):
            self.register_parameter('Z', Parameter(torch.ones(rank) / rank))
        else:
            self.register_parameter('Z', None)

        if hasattr(self, "Z"):
            self.Z.data.div_(get_norm(self.Z))
            infer_rank = self.Z.shape[0]

        if infer_rank is None:
            assert rank, "A rank should be given when W, H and Z are not available!"
        else:
            if hasattr(self, "Z"):
                assert self.Z.shape[0] == infer_rank, "Latent size Z does not match with others!"
            if hasattr(self, "H"):
                assert self.H.shape[1] == infer_rank, "Latent size H does not match with others!"
            if hasattr(self, "W"):
                assert self.W.shape[1] == infer_rank, "Latent size W does not match with others!"
            rank = infer_rank

        self.rank = rank

    def extra_repr(self) -> str:
        s = ('{rank}')
        if self.W is not None:
            s += ', out_channels={out_channels}'
            if hasattr(self, 'kernel_size'):
                s += ', kernel_size={kernel_size}'
        return s.format(**self.__dict__)

    def forward(self, H: Tensor = None, W: Tensor = None, Z: Tensor = None, norm: float = None) -> Tensor:
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        if Z is None:
            Z = self.Z

        result = self.reconstruct(W, Z, H)
        if norm is None:
            return result
        return result * norm

    @staticmethod
    def reconstruct(H: Tensor, W: Tensor, Z: Tensor) -> Tensor:
        r"""Defines the computation performed at every call.

            Should be overridden by all subclasses.
            """
        raise NotImplementedError

    def fit(self,
            V: Tensor,
            tol: float = 1e-4,
            max_iter: int = 200,
            verbose: bool = False,
            W_alpha: float = 1,
            Z_alpha: float = 1,
            H_alpha: float = 1):

        assert torch.all(V >= 0.), "Target should be non-negative."
        W = self.W
        H = self.H
        Z = self.Z

        norm = get_norm(V)
        V = V / norm

        with torch.no_grad():
            WZH = self.reconstruct(H, W, Z)
            loss_init = previous_loss = kl_div(WZH, V).mul(2).sqrt().item()

        with tqdm(total=max_iter, disable=not verbose) as pbar:
            for n_iter in range(max_iter):
                self.zero_grad()
                WZH = self.reconstruct(H, W, Z)
                WZH.backward(V / WZH.add(eps))

                Z_prior = None
                if Z.requires_grad:
                    Z.data.mul_(Z.grad.relu())
                    Z_prior = Z.detach()

                if W.requires_grad:
                    W.data.mul_(W.grad.relu())
                    if Z_prior is None:
                        W_divider = get_norm(W)
                        Z_prior = W_divider.squeeze()
                    else:
                        W_divider = Z_prior[(
                            slice(None),) + (None,) * (W.dim() - 2)]
                    W.data.div_(W_divider)

                if H.requires_grad:
                    H.data.mul_(H.grad.relu())
                    if Z_prior is None:
                        H_divider = get_norm(H)
                    else:
                        H_divider = Z_prior[(
                            slice(None),) + (None,) * (H.dim() - 2)]
                    W.data.div_(H_divider)

                if Z_alpha != 1:
                    Z.data.add_(Z_alpha - 1).relu_()
                if W_alpha != 1:
                    W.data.add_(W_alpha - 1).relu_()
                if H_alpha != 1:
                    H.data.add_(H_alpha - 1).relu_()

                if Z.requires_grad:
                    Z.data.div_(get_norm(Z))
                if W.requires_grad:
                    W.data.div_(get_norm(W))
                if H.requires_grad:
                    H.data.div_(get_norm(H))

                if n_iter % 10 == 9:
                    with torch.no_grad():
                        WZH = self.reconstruct(H, W, Z)
                        loss = kl_div(WZH, V).mul(2).sqrt().item()
                        log_pro = _log_probability(
                            V, WZH, W, Z, H, W_alpha, Z_alpha, H_alpha).item()
                    pbar.set_postfix(loss=loss, log_likelihood=log_pro)
                    pbar.update(10)
                    if (previous_loss - loss) / loss_init < tol:
                        break
                    previous_loss = loss

        return n_iter, norm


class PLCA(BaseComponent):

    def __init__(self,
                 Vshape: Iterable[int] = None,
                 rank: int = None,
                 **kwargs):
        if isinstance(Vshape, Iterabc):
            M, K = Vshape
            rank = rank if rank else K
            kwargs['W'] = (K, rank)
            kwargs['H'] = (M, rank)

        super().__init__(rank, **kwargs)

    @staticmethod
    def reconstruct(H, W, Z):
        return torch.einsum("bi,ji,i->bj", H, W, Z)


class SIPLCA(BaseComponent):

    def __init__(self,
                 Vshape: Iterable[int] = None,
                 rank: int = None,
                 T: _size_1_t = 1,
                 **kwargs):
        if isinstance(Vshape, Iterabc):
            T, = _single(T)
            batch, K, M = Vshape
            rank = rank if rank else K
            kwargs['W'] = (K, rank, T)
            kwargs['H'] = (batch, rank, M - T + 1)

        super().__init__(rank, **kwargs)

    @staticmethod
    def reconstruct(H, W, Z):
        pad_size = W.shape[2] - 1
        return F.conv1d(H, W.flip(2) * Z.view(-1, 1), padding=pad_size)


class SIPLCA2(BaseComponent):

    def __init__(self,
                 Vshape: Iterable[int] = None,
                 rank: int = None,
                 kernel_size: _size_2_t = 1,
                 **kwargs):
        if isinstance(Vshape, Iterabc):
            kernel_size = _pair(kernel_size)
            H, W = kernel_size
            batch, channel, K, M = Vshape
            rank = rank if rank else K
            kwargs['W'] = (channel, rank,) + kernel_size
            kwargs['H'] = (batch, rank, K - H + 1, M - W + 1)
        super().__init__(rank, **kwargs)

    @staticmethod
    def reconstruct(H, W, Z):
        pad_size = (W.shape[2] - 1, W.shape[3] - 1)
        out = F.conv2d(H, W.flip((2, 3)) * Z.view(-1, 1, 1), padding=pad_size)
        return out


class SIPLCA3(BaseComponent):
    def __init__(self,
                 Vshape: Iterable[int] = None,
                 rank: int = None,
                 kernel_size: _size_3_t = 1,
                 **kwargs):
        if isinstance(Vshape, Iterabc):
            kernel_size = _triple(kernel_size)
            D, H, W = kernel_size
            batch, channel, N, K, M = Vshape
            rank = rank if rank else K
            kwargs['W'] = (channel, rank) + kernel_size
            kwargs['H'] = (batch, rank, N - D + 1, K - H + 1, M - W + 1)

        super().__init__(rank, **kwargs)

    @staticmethod
    def reconstruct(H, W, Z):
        pad_size = (W.shape[2] - 1, W.shape[3] - 1, W.shape[4] - 1)
        out = F.conv3d(H, W.flip((2, 3, 4)) *
                       Z.view(-1, 1, 1, 1), padding=pad_size)
        return out
