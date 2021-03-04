import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from typing import Union, Iterable, Optional, List, Tuple
from collections.abc import Iterable as Iterabc
from .metrics import beta_div
from tqdm import tqdm

_size_1_t = Union[int, Tuple[int]]
_size_2_t = Union[int, Tuple[int, int]]
_size_3_t = Union[int, Tuple[int, int, int]]

__all__ = [
    'BaseComponent', 'NMF', 'NMFD', 'NMF2D', 'NMF3D'
]

eps = 1e-8


def _proj_func(s: Tensor,
               k1: float,
               k2: float) -> Tensor:
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


def _double_backward_update(V: Tensor,
                            WH: Tensor,
                            param: Parameter,
                            beta: float,
                            gamma: float,
                            l1_reg: float,
                            l2_reg: float,
                            pos: Tensor = None):
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
    neg = torch.clone(param.grad).relu_().add_(eps)

    if pos is None:
        param.grad.zero_()
        WH.backward(output_pos)
        pos = torch.clone(param.grad).relu_().add_(eps)

    if l1_reg > 0:
        pos.add_(l1_reg)
    if l2_reg > 0:
        pos.add_(param.data, alpha=l2_reg)
    multiplier = neg / pos
    if gamma != 1:
        multiplier.pow_(gamma)
    param.data.mul_(multiplier)


def _get_W_kl_positive(H: Tensor) -> Tensor:
    sum_dims = list(range(H.dim()))
    sum_dims.remove(1)
    return H.sum(sum_dims, keepdims=True)


def _get_H_kl_positive(W: Tensor) -> Tensor:
    sum_dims = list(range(W.dim()))
    sum_dims.remove(1)
    return W.sum(sum_dims, keepdims=True).squeeze(0)


def _get_norm(x: Tensor,
              axis: int = 1) -> Tensor:
    x2 = x * x
    sum_dims = list(range(x2.dim()))
    sum_dims.remove(axis)
    return x2.sum(sum_dims).sqrt()


@torch.no_grad()
def _renorm(W: Tensor,
            H: Tensor,
            unit_norm='W'):
    if unit_norm == 'W':
        W_norm = _get_norm(W)
        slicer = (slice(None),) + (None,) * (W.dim() - 2)
        W /= W_norm[slicer]
        slicer = (slice(None),) + (None,) * (H.dim() - 2)
        H *= W_norm[slicer]
    elif unit_norm == 'H':
        H_norm = _get_norm(H)
        slicer = (slice(None),) + (None,) * (H.dim() - 2)
        H /= H_norm[slicer]
        slicer = (slice(None),) + (None,) * (W.dim() - 2)
        W *= H_norm[slicer]
    else:
        raise ValueError("Input type isn't valid!")


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
                       'out_channels': Optional[int],
                       'kernel_size': Optional[Tuple[int, ...]]}

    rank: int
    W: Optional[Tensor]
    H: Optional[Tensor]
    out_channels: Optional[int]
    kernel_size: Optional[Tuple[int, ...]]

    def __init__(self,
                 rank: int = None,
                 W: Union[Iterable[int], Tensor] = None,
                 H: Union[Iterable[int], Tensor] = None,
                 trainable_W: bool = True,
                 trainable_H: bool = True):
        super().__init__()

        if isinstance(W, Tensor):
            self.register_parameter('W', Parameter(
                torch.empty(*W.size()), requires_grad=trainable_W))
            self.W.data.copy_(W)
        elif isinstance(W, Iterabc) and trainable_W:
            self.register_parameter('W', Parameter(torch.randn(*W).abs()))
        else:
            self.register_parameter('W', None)

        if isinstance(H, Tensor):
            H_shape = H.shape
            self.register_parameter('H', Parameter(
                torch.empty(*H_shape), requires_grad=trainable_H))
            self.H.data.copy_(H)
        elif isinstance(H, Iterabc) and trainable_H:
            self.register_parameter('H', Parameter(torch.randn(*H).abs()))
        else:
            self.register_parameter('H', None)

        if isinstance(self.W, Tensor):
            if isinstance(self.H, Tensor):
                assert self.W.shape[1] == self.H.shape[1], "Latent size of W and H should be equal!"
            rank = self.W.shape[1]
            self.out_channels = self.W.shape[0]
            if len(self.W.shape) > 2:
                self.kernel_size = tuple(self.W.shape[2:])
        elif isinstance(self.H, Tensor):
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

    def forward(self, H: Tensor = None, W: Tensor = None) -> Tensor:
        r"""An outer wrapper of :meth:`self.reconstruct(H,W) <torchnmf.nmf.BaseComponent.reconstruct>`.

        .. note::
                Should call the :class:`BaseComponent` instance afterwards
                instead of this since the former takes care of running the
                registered hooks while the latter silently ignores them.

        Args:
            H(Tensor, optional): input activation tensor H. If no tensor was given will use :attr:`H` from this module
                                instead.
            W(Tensor, optional): input template tensor W. If no tensor was given will use :attr:`W` from this module
                                instead.

        Returns:
            tensor
        """
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        return self.reconstruct(H, W)

    @staticmethod
    def reconstruct(H: Tensor, W: Tensor) -> Tensor:
        r"""Defines the computation performed at every call.

            Should be overridden by all subclasses.
            """
        raise NotImplementedError

    def fit(self,
            V: Tensor,
            beta: float = 1,
            tol: float = 1e-4,
            max_iter: int = 200,
            verbose: bool = False,
            alpha: float = 0,
            l1_ratio: float = 0
            ) -> int:
        r"""Learn a NMF model for the data V by minimizing beta divergence.

        To invoke this function, attributes :meth:`H <torchnmf.nmf.BaseComponent.H>` and
        :meth:`W <torchnmf.nmf.BaseComponent.H>` should be presented in this module.

        Args:
            V (Tensor): data tensor to be decomposed.
            beta (float): beta divergence to be minimized, measuring the distance between V and the NMF model.
                        Default: ``1.``.
            tol (float): tolerance of the stopping condition. Default: ``1e-4``.
            max_iter (int): maximum number of iterations before timing out. Default: ``200``.
            verbose (bool): whether to be verbose. Default: ``False``.
            alpha (float): constant that multiplies the regularization terms. Set it to zero to have no regularization.
                            Default: ``0``.
            l1_ratio (float):  the regularization mixing parameter, with 0 <= l1_ratio <= 1.
                                For l1_ratio = 0 the penalty is an elementwise L2 penalty (aka Frobenius Norm).
                                For l1_ratio = 1 it is an elementwise L1 penalty.
                                For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2. Default: ``0``.

        Returns:
            total number of iterations.
        """

        W = self.W
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
            loss_init = previous_loss = beta_div(
                WH, V, beta).mul(2).sqrt().item()

        with tqdm(total=max_iter, disable=not verbose) as pbar:
            for n_iter in range(max_iter):
                if W.requires_grad:
                    WH = self.reconstruct(H.detach(), W)
                    _double_backward_update(V, WH, W, beta, gamma, l1_reg, l2_reg,
                                            _get_W_kl_positive(H.detach()) if beta == 1 else None)

                if H.requires_grad:
                    WH = self.reconstruct(H, W.detach())
                    _double_backward_update(V, WH, H, beta, gamma, l1_reg, l2_reg,
                                            _get_H_kl_positive(W.detach()) if beta == 1 else None)

                if n_iter % 10 == 9:
                    with torch.no_grad():
                        WH = self.reconstruct(H, W)
                        loss = beta_div(WH, V, beta).mul(2).sqrt().item()
                    pbar.set_postfix(loss=loss)
                    pbar.update(10)
                    if (previous_loss - loss) / loss_init < tol:
                        break
                    previous_loss = loss

        return n_iter

    def sparse_fit(self,
                   V,
                   beta=1,
                   max_iter=200,
                   verbose=0,
                   sW=None,
                   sH=None,
                   ) -> int:
        r"""Learn a NMF model for the data V by minimizing beta divergence with sparseness constraints proposed in
        `Non-negative Matrix Factorization with Sparseness Constraints`_.

        To invoke this function, attributes :meth:`H <torchnmf.nmf.BaseComponent.H>` and
        :meth:`W <torchnmf.nmf.BaseComponent.H>` should be presented in this module.

        .. _`Non-negative Matrix Factorization with Sparseness Constraints`:
            https://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf

        Args:
            V (Tensor): data tensor to be decomposed.
            beta (float): beta divergence to be minimized, measuring the distance between V and the NMF model.
                        Default: ``1.``.
            max_iter (int): maximum number of iterations before timing out. Default: ``200``.
            verbose (bool): whether to be verbose. Default: ``False``.
            sW (float or None): the target sparseness for template tensor :attr:`W` , with 0 < sW < 1. Set it to ``None``
                will have no constraint. Default: ``None``
            sH (float or None): the target sparseness for activation tensor :attr:`H` , with 0 < sH < 1. Set it to ``None``
                will have no constraint. Default: ``None``

        Returns:
            total number of iterations.
        """
        W = self.W
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
                                                _get_W_kl_positive(H.detach()) if beta == 1 else None)
                    else:
                        W.grad = None
                        WH = self.reconstruct(H.detach(), W)
                        loss = beta_div(WH, V, beta)
                        loss.backward()
                        with torch.no_grad():
                            for i in range(10):
                                Wnew = W - stepsize_W * W.grad
                                norms = _get_norm(Wnew)
                                for j in range(Wnew.shape[1]):
                                    Wnew[:, j] = _proj_func(
                                        Wnew[:, j], L1a * norms[j], norms[j] ** 2)
                                new_loss = beta_div(self.reconstruct(self.H, Wnew),
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
                                                _get_H_kl_positive(W.detach()) if beta == 1 else None)
                    else:
                        H.grad = None
                        WH = self.reconstruct(H, W.detach())
                        loss = beta_div(WH, V, beta)
                        loss.backward()

                        with torch.no_grad():
                            for i in range(10):
                                Hnew = H - stepsize_H * H.grad
                                norms = _get_norm(Hnew)
                                for j in range(H.shape[1]):
                                    Hnew[:, j] = _proj_func(
                                        Hnew[:, j], L1s * norms[j], norms[j] ** 2)
                                new_loss = beta_div(self.reconstruct(Hnew, W),
                                                    V, beta)
                                if new_loss <= loss:
                                    break

                                stepsize_H *= 0.5

                            stepsize_H *= 1.2
                            H.copy_(Hnew)
                    _renorm(W, H, 'H')

                if n_iter % 10 == 9:
                    with torch.no_grad():
                        WH = self.reconstruct(H, W)
                        loss = beta_div(WH, V, beta).mul(2).sqrt().item()
                    pbar.set_postfix(loss=loss)
                    pbar.update(10)
        return n_iter


class NMF(BaseComponent):
    r"""Non-Negative Matrix Factorization (NMF).

    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix V: :math:`V \approx WH`

    This factorization can be used for example for dimensionality reduction, source separation or topic extraction.

    Note:
        To match with PyTorch convention, this class actually use :math:`H^T` so the batch dimension is the first
        dimension, and require input target matrix V is also transposed.

    Args:
        Vshape (tuple, optional): Size of target matrix V
        rank (int, optional): Size of hidden dimension
        **kwargs: Arguments passed through to :meth:`BaseComponent <torchnmf.nmf.BaseComponent>`.


    Shape:
        - V: :math:`(N, C)`
        - W: :math:`(C, R)`
        - H: :math:`(N, R)`


    Examples::

        >>> V = torch.rand(30, 20)
        >>> m = NMF(V.t().shape, 5)
        >>> m.W.size()
        torch.Size([30, 5])
        >>> m.H.size()
        torch.Size([20, 5])
        >>> WHt = m()
        >>> WHt.size()
        torch.Size([20, 30])

    """

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
    def reconstruct(H, W):
        return F.linear(H, W)


class NMFD(BaseComponent):
    r"""Non-negative Matrix Factor Deconvolution (NMFD).

    Find non-negative matrix H and 3-dimensional tensor W whose convolutional output approximates the non-
    negative matrix V:

    .. math::
        \mathbf{V} \approx \sum_{t=0}^{T-1} \mathbf{W}_{t} \cdot \stackrel{t \rightarrow}{\mathbf{H}}

    More precisely:

    .. math::
        V_{i,j} \approx \sum_{t=0}^{T-1} \sum_{r=0}^{\text{rank}-1}
        W_{i,r,t} * H_{r, j - t}

    Look at the paper:
    `Non-negative Matrix Factor Deconvolution; Extraction of Multiple Sound Sources from Monophonic Inputs`_
    by Paris Smaragdis (2004) for more details.

    Note:
        To match with PyTorch convention, an extra batch dimension is required for target matrix V.

    Args:
        Vshape (tuple, optional): Size of target matrix V
        rank (int, optional): Size of hidden dimension
        T (int, optional): Size of the convolving window
        **kwargs: Arguments passed through to :meth:`BaseComponent <torchnmf.nmf.BaseComponent>`.


    Shape:
        - V: :math:`(N, C, L_{out})`
        - W: :math:`(C, R, T)`
        - H: :math:`(N, R, L_{in})` where

        .. math::
            L_{in} = L_{out} - T + 1

    Examples::

        >>> V = torch.rand(33, 50).unsqueeze(0)
        >>> m = NMF(V.shape, 16, 3)
        >>> m.W.size()
        torch.Size([33, 16, 3])
        >>> m.H.size()
        torch.Size([1, 16, 48])
        >>> WHt = m()
        >>> WHt.size()
        torch.Size([1, 33, 50])

    .. _Non-negative Matrix Factor Deconvolution; Extraction of Multiple Sound Sources from Monophonic Inputs:
        https://www.math.uci.edu/icamp/summer/research_11/esser/nmfaudio.pdf

    """

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
    def reconstruct(H, W):
        pad_size = W.shape[2] - 1
        return F.conv1d(H, W.flip(2), padding=pad_size)


class NMF2D(BaseComponent):
    r"""Nonnegative Matrix Factor 2-D Deconvolution (NMF2D).

    Find non-negative 3-dimensional tensor H and 4-dimensional tensor W whose 2D convolutional output
    approximates the non-negative 3-dimensional tensor V:

    .. math::
        \mathbf{V} \approx \sum_{\tau} \sum_{\phi} \stackrel{\downarrow \phi}{\mathbf{W}^{\tau}}
        \stackrel{\rightarrow \tau}{\mathbf{H}^{\phi}}

    More precisely:

    .. math::
        V_{i,j,k} \approx \sum_{l=0}^{k_1-1} \sum_{m=0}^{k_2-1} \sum_{r=0}^{\text{rank}-1}
        W_{i,r,l,m} * H_{r, j-l,k-m}

    Look at the paper:
    `Nonnegative Matrix Factor 2-D Deconvolution for Blind Single Channel Source Separation`_
    by Schmidt et al. (2006) for more details.

    Note:
        To match with PyTorch convention, an extra batch dimension is required for target tensor V.

    Args:
        Vshape (tuple, optional): Size of target tensor V
        rank (int, optional): Size of hidden dimension
        kernel_size (int or tuple, optional): Size of the convolving kernel
        **kwargs: Arguments passed through to :meth:`BaseComponent <torchnmf.nmf.BaseComponent>`.

    Shape:
        - V: :math:`(N, C, L_{out}, M_{out})`
        - W: :math:`(C, R, \text{kernel_size}[0], \text{kernel_size}[1])`
        - H: :math:`(N, R, L_{in}, M_{in})` where

        .. math::
            L_{in} = L_{out} - \text{kernel_size}[0] + 1
        .. math::
            M_{in} = M_{out} - \text{kernel_size}[1] + 1

    Examples::

        >>> V = torch.rand(33, 50).unsqueeze(0).unsqueeze(0)
        >>> m = NMF2D(V.shape, 16, 3)
        >>> m.W.size()
        torch.Size([1, 16, 3, 3])
        >>> m.H.size()
        torch.Size([1, 16, 31, 48])
        >>> WHt = m()
        >>> WHt.size()
        torch.Size([1, 1, 33, 50])

    .. _Nonnegative Matrix Factor 2-D Deconvolution for Blind Single Channel Source Separation:
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.422.6689&rep=rep1&type=pdf

        """

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
    def reconstruct(H, W):
        pad_size = (W.shape[2] - 1, W.shape[3] - 1)
        out = F.conv2d(H, W.flip((2, 3)), padding=pad_size)
        return out


class NMF3D(BaseComponent):
    r"""Nonnegative Matrix Factor 3-D Deconvolution (NMF3D).

    Find non-negative 4-dimensional tensor H and 5-dimensional tensor W whose 2D convolutional output
    approximates the non-negative 4-dimensional tensor V:

    .. math::
        V_{i,j,k,l} \approx \sum_{m=0}^{k_1-1} \sum_{n=0}^{k_2-1} \sum_{u=0}^{k_3-1} \sum_{r=0}^{\text{rank}-1}
        W_{i,r,m,n,u} * H_{r,j-m,k-n,l-u}

    Note:
        To match with PyTorch convention, an extra batch dimension is required for target tensor V.

    Args:
        Vshape (tuple, optional): Size of target tensor V
        rank (int, optional): Size of hidden dimension
        kernel_size (int or tuple, optional): Size of the convolving kernel
        **kwargs: Arguments passed through to :meth:`BaseComponent <torchnmf.nmf.BaseComponent>`.


    Shape:
        - V: :math:`(N, C, L_{out}, M_{out}, O_{out})`
        - W: :math:`(C, R, \text{kernel_size}[0], \text{kernel_size}[1], \text{kernel_size}[2])`
        - H: :math:`(N, R, L_{in}, M_{in}, O_{in})` where

        .. math::
            L_{in} = L_{out} - \text{kernel_size}[0] + 1
        .. math::
            M_{in} = M_{out} - \text{kernel_size}[1] + 1
        .. math::
            O_{in} = O_{out} - \text{kernel_size}[2] + 1

    Examples::

        >>> V = torch.rand(3, 64, 64, 100).unsqueeze(0)
        >>> m = NMF3D(V.shape, 8, (5, 5, 20))
        >>> m.W.size()
        torch.Size([3, 8, 5, 5, 20])
        >>> m.H.size()
        torch.Size([1, 8, 60, 60, 81])
        >>> WHt = m()
        >>> WHt.size()
        torch.Size([1, 3, 64, 64, 100])

    """

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
    def reconstruct(H, W):
        pad_size = (W.shape[2] - 1, W.shape[3] - 1, W.shape[4] - 1)
        out = F.conv3d(H, W.flip((2, 3, 4)), padding=pad_size)
        return out
