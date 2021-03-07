import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from .base import Base
from .utils import normalize
from .nmf import _get_H_kl_positive, _get_W_kl_positive
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


def _double_backward_update(V: Tensor,
                            WZH: Tensor,
                            param: Parameter,
                            pos: Tensor = None,
                            retain_graph=False):
    param.grad = None
    # first backward
    WH.backward(V / WZH, retain_graph=retain_graph)
    neg = torch.clone(param.grad).relu_().add_(eps)
    multiplier = neg / pos
    param.data.mul_(multiplier)


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
        elif isinstance(W, Iterabc) and trainable_W:
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
        elif isinstance(H, Iterabc) and trainable_H:
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

    def forward(self, H: Tensor = None, W: Tensor = None, Z: Tensor = None) -> Tensor:
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        if Z is None:
            Z = self.Z
        return self.reconstruct(W, Z, H)

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

                if Z.requires_grad:
                    Z.data.mul_(Z.grad.relu())

                if W.requires_grad:
                    W.data.mul_(W.grad.relu())
                    W.data.div_(Z if Z.requires_grad else get_norm(W))

                if H.requires_grad:
                    H.data.mul_(H.grad.relu())
                    H.data.div_(Z if Z.requires_grad else get_norm(W))

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


class PLCA(_PLCA):

    def __init__(self, Vshape: tuple, rank: int = None, uniform=False):
        self.K, self.M = Vshape
        if not rank:
            rank = self.K
        super().__init__((self.K, rank), rank, (rank, self.M), (0,), (1,), uniform)

    def reconstruct(self, W, Z, H):
        return (W * Z) @ H

    def update_params(self, VdivWZH, update_W, update_H, update_Z, W_alpha, H_alpha, Z_alpha):
        # type: (Tensor, bool, bool, bool, float, float, float) -> None
        if update_W or update_Z:
            new_W = VdivWZH @ self.H.t() * self.W * self.Z

        if update_H:
            H = self.fix_neg(self.W.mul(self.Z).t() @
                             VdivWZH * self.H + H_alpha - 1)
            self.H[:] = normalize(H, 1)

        if update_W:
            self.W[:] = normalize(self.fix_neg(new_W + W_alpha - 1), 0)

        if update_Z:
            self.Z[:] = normalize(self.fix_neg(new_W.sum(0) + Z_alpha - 1))

    def sort(self):
        _, maxidx = self.W.data.max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        self.H.data = self.H.data[idx]
        self.Z.data = self.Z.data[idx]


class SIPLCA(_PLCA):

    def __init__(self, Vshape: tuple, rank: int = None, T: int = 1, uniform=False):
        self.K, self.M = Vshape
        self.pad_size = T - 1
        if not rank:
            rank = self.K

        W_size = (self.K, rank, T)
        H_size = (rank, self.M - T + 1)
        H_norm_dim = (1,)
        super().__init__(W_size, rank, H_size, (0, 2), H_norm_dim, uniform)

    def reconstruct(self, W, Z, H):
        return F.conv1d(H[None, ...], W.flip(2) * Z[:, None], padding=self.pad_size)[0]

    def update_params(self, VdivWZH, update_W, update_H, update_Z, W_alpha, H_alpha, Z_alpha):
        # type: (Tensor, bool, bool, bool, float, float, float) -> None

        if update_W or update_Z:
            new_W = F.conv1d(VdivWZH[:, None], self.H[:, None]
                             * self.Z[:, None, None]) * self.W

        if update_H:
            new_H = F.conv1d(VdivWZH[None, ...], torch.transpose(
                self.W * self.Z[:, None], 0, 1))[0] * self.H
            new_H = normalize(self.fix_neg(new_H + H_alpha - 1), 1)
            self.H[:] = new_H

        if update_W:
            self.W[:] = normalize(self.fix_neg(new_W + W_alpha - 1), (0, 2))

        if update_Z:
            Z = normalize(self.fix_neg(new_W.sum((0, 2)) + Z_alpha - 1))
            self.Z[:] = Z

    def sort(self):
        _, maxidx = self.W.data.sum(2).max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        self.H.data = self.H.data[idx]
        self.Z.data = self.Z.data[idx]


class SIPLCA2(_PLCA):

    def __init__(self, Vshape: tuple, rank: int = None, win=1, uniform=False):
        try:
            F, T = win
        except:
            F = T = win
        if len(Vshape) == 3:
            self.channel, self.K, self.M = Vshape
        else:
            self.K, self.M = Vshape
            self.channel = 1

        self.pad_size = (F - 1, T - 1)
        if not rank:
            rank = self.K

        W_size = (self.channel, rank, F, T)
        H_size = (rank, self.K - F + 1, self.M - T + 1)
        W_norm_dim = (0, 2, 3)
        H_norm_dim = (1, 2)
        super().__init__(W_size, rank, H_size, W_norm_dim, H_norm_dim, uniform)

    def reconstruct(self, W, Z, H):
        out = F.conv2d(H[None, ...], W.mul(Z[:, None, None]).flip(
            (2, 3)), padding=self.pad_size)[0]
        if self.channel == 1:
            return out[0]
        return out

    def update_params(self, VdivWZH, update_W, update_H, update_Z, W_alpha, H_alpha, Z_alpha):
        # type: (Tensor, bool, bool, bool, float, float, float) -> None
        VdivWZH = VdivWZH.view(self.channel, 1, self.K, self.M)
        if update_W or update_Z:
            new_W = F.conv2d(VdivWZH, self.H.mul(
                self.Z[:, None, None])[:, None]) * self.W

        if update_H:
            new_H = F.conv2d(VdivWZH.transpose(0, 1), torch.transpose(
                self.W * self.Z[:, None, None], 0, 1))[0] * self.H
            new_H = normalize(self.fix_neg(new_H + H_alpha - 1), (1, 2))
            self.H[:] = new_H

        if update_W:
            self.W[:] = normalize(self.fix_neg(new_W + W_alpha - 1), (0, 2, 3))

        if update_Z:
            Z = normalize(self.fix_neg(new_W.sum((0, 2, 3)) + Z_alpha - 1))
            self.Z[:] = Z

    def sort(self):
        raise NotImplementedError


class SIPLCA3(_PLCA):
    def __init__(self, Vshape: tuple, rank: int = None, win=1, uniform=False):
        try:
            T, H, W = win
        except:
            T = H = W = win
        if len(Vshape) == 4:
            self.channel, self.N, self.K, self.M = Vshape
        else:
            self.N, self.K, self.M = Vshape
            self.channel = 1

        self.pad_size = (T - 1, H - 1, W - 1)
        if not rank:
            rank = self.K

        W_size = (self.channel, rank, T, H, W)
        H_size = (rank, self.N - T + 1, self.K - H + 1, self.M - W + 1)
        W_norm_dim = (0, 2, 3, 4)
        H_norm_dim = (1, 2, 3)
        super().__init__(W_size, rank, H_size, W_norm_dim, H_norm_dim, uniform)

    def reconstruct(self, W, Z, H):
        out = F.conv3d(H[None, ...], W.mul(Z[:, None, None, None]).flip(
            (2, 3, 4)), padding=self.pad_size)[0]
        if self.channel == 1:
            return out[0]
        return out

    def update_params(self, VdivWZH, update_W, update_H, update_Z, W_alpha, H_alpha, Z_alpha):
        # type: (Tensor, bool, bool, bool, float, float, float) -> None
        VdivWZH = VdivWZH.view(self.channel, 1, self.N, self.K, self.M)
        if update_W or update_Z:
            new_W = F.conv3d(VdivWZH, self.H.mul(self.Z[:, None, None, None])[
                             :, None], padding=self.pad_size) * self.W

        if update_H:
            new_H = F.conv3d(VdivWZH.transpose(0, 1), torch.transpose(self.W * self.Z[:, None, None, None], 0, 1))[
                0] * self.H
            new_H = normalize(self.fix_neg(new_H + H_alpha - 1), (1, 2, 3))
            self.H[:] = new_H

        if update_W:
            self.W[:] = normalize(self.fix_neg(
                new_W + W_alpha - 1), (0, 2, 3, 4))

        if update_Z:
            Z = normalize(self.fix_neg(new_W.sum((0, 2, 3, 4)) + Z_alpha - 1))
            self.Z[:] = Z

    def sort(self):
        raise NotImplementedError
