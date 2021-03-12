import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter
from .nmf import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple
from tqdm import tqdm
from typing import Union, Iterable, Optional, List, Tuple
from collections.abc import Iterable as Iterabc
from .metrics import kl_div
from .constants import eps

__all__ = [
    'PLCA', 'SIPLCA', 'SIPLCA2', 'SIPLCA3', 'BaseComponent'
]


def _log_probability(V, WZH, W, Z, H, W_alpha, Z_alpha, H_alpha):
    return V.view(-1) @ WZH.log().view(-1) + W.log().mul(W_alpha - 1).sum() + H.log().mul(
        H_alpha - 1).sum() + Z.log().mul(Z_alpha - 1).sum()


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
    r"""Base class for all PLCA modules.

    You can't use this module directly.
    Your models should also subclass this class.

    Args:
        rank (int): size of hidden dimension
        W (tuple or Tensor): size or initial probabilities of template tensor W
        H (tuple or Tensor): size or initial probabilities of activation tensor H
        Z (Tensor): initial probabilities of latent vector Z
        trainable_W (bool):  controls whether template tensor W is trainable when initial probabilities is given. Default: ``True``
        trainable_H (bool):  controls whether activation tensor H is trainable when initial probabilities is given. Default: ``True``
        trainable_Z (bool):  controls whether latent vector Z is trainable when initial probabilities is given. Default: ``True``


    Attributes:
        W (Tensor or None): the template tensor of the module if corresponding argument is given.
            If size is given, values are initialized randomly.
        H (Tensor or None): the activation tensor of the module if corresponding argument is given.
            If size is given, values are initialized randomly.
        Z (Tensor or None): the latent vector of the module if corresponding argument or rank is given.
            If rank is given, values are initialized uniformly.

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

        if getattr(self, "W") is not None:
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

        if getattr(self, "H") is not None:
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

        if getattr(self, "Z") is not None:
            self.Z.data.div_(get_norm(self.Z))
            infer_rank = self.Z.shape[0]

        if infer_rank is None:
            assert rank, "A rank should be given when W, H and Z are not available!"
        else:
            if getattr(self, "Z") is not None:
                assert self.Z.shape[0] == infer_rank, "Latent size of Z does not match with others!"
            if getattr(self, "H") is not None:
                assert self.H.shape[1] == infer_rank, "Latent size of H does not match with others!"
            if getattr(self, "W") is not None:
                assert self.W.shape[1] == infer_rank, "Latent size of W does not match with others!"
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
        r"""An outer wrapper of :meth:`self.reconstruct(H,W,Z) <torchnmf.plca.BaseComponent.reconstruct>`.

        .. note::
                Should call the :class:`BaseComponent` instance afterwards
                instead of this since the former takes care of running the
                registered hooks while the latter silently ignores them.

        Args:
            H(Tensor, optional): input activation tensor H. If no tensor was given will use :attr:`H` from this module
                                instead
            W(Tensor, optional): input template tensor W. If no tensor was given will use :attr:`W` from this module
                                instead
            Z(Tensor, optional): input latent vector Z. If no tensor was given will use :attr:`Z` from this module
                                instead
            norm(float, optional): a scaling value multiply on output before return. Default: ``1``

        Returns:
            Tensor: tensor
        """
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        if Z is None:
            Z = self.Z

        result = self.reconstruct(H, W, Z)
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
            W_alpha: Union[float, Tensor] = 1.,
            H_alpha: Union[float, Tensor] = 1.,
            Z_alpha: Union[float, Tensor] = 1.):
        r"""Learn a PLCA model for the data V by maximizing the following log probability of V 
        and model params :math:`\theta` using EM algorithm:

        .. math::
            \mathcal{L} (\theta)= \sum_{k_1...k_N} v_{k_1...k_N}\log{\hat{v}_{k_1...k_N}} \\
            + \sum_k (\alpha_{z,k} - 1) \log z_k  \\
            + \sum_{f_1...f_M} (\alpha_{w,f_1...f_M} - 1) \log w_{f_1...f_M} \\
            + \sum_{\tau_1...\tau_L} (\alpha_{h,\tau_1...\tau_L} - 1) \log h_{\tau_1...\tau_L} \\

        Where :math:`\hat{V}` is the reconstructed output, N is the number of dimensions of target tensor :math:`V`,
        M is the number of dimensions of tensor :math:`W`, and L is the number of dimensions of tensor :math:`H`.
        The last three terms come from Dirichlet prior assumption.

        To invoke this function, attributes :meth:`H <torchnmf.plca.BaseComponent.H>`,
        :meth:`W <torchnmf.plca.BaseComponent.H>` and :meth:`Z <torchnmf.plca.BaseComponent.Z>`
        should be presented in this module.

        Args:
            V (Tensor): data tensor to be decomposed
            tol (float): tolerance of the stopping condition. Default: ``1e-4``
            max_iter (int): maximum number of iterations before timing out. Default: ``200``
            verbose (bool): whether to be verbose. Default: ``False``
            W_alpha (float): hyper parameter of Dirichlet prior on W. 
                            Can be a scalar or a tensor that is broadcastable to W. 
                            Set it to one to have no regularization. Default: ``1``
            H_alpha (float): hyper parameter of Dirichlet prior on H. 
                            Can be a scalar or a tensor that is broadcastable to H. 
                            Set it to one to have no regularization. Default: ``1``
            Z_alpha (float): hyper parameter of Dirichlet prior on Z. 
                            Can be a scalar or a tensor that is broadcastable to Z. 
                            Set it to one to have no regularization. Default: ``1``
        Returns:
            tuple: a length-2 tuple with first element is total number of iterations, and the second is the sum of tensor V
        """

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
                    Z_prior = Z.clone()
                    if Z_alpha != 1:
                        Z.data.add_(Z_alpha - 1).relu_()
                    Z.data.div_(Z.sum())

                if W.requires_grad:
                    W.data.mul_(W.grad.relu())
                    if Z_prior is None:
                        W_divider = get_norm(W)
                        Z_prior = W_divider.squeeze()
                    else:
                        W_divider = Z_prior[(
                            slice(None),) + (None,) * (W.dim() - 2)]
                    W.data.div_(W_divider)
                    if W_alpha != 1:
                        W.data.add_(W_alpha - 1).relu_()
                        W.data.div_(get_norm(W))

                if H.requires_grad:
                    H.data.mul_(H.grad.relu())
                    if Z_prior is None:
                        H_divider = get_norm(H)
                    else:
                        H_divider = Z_prior[(
                            slice(None),) + (None,) * (H.dim() - 2)]
                    W.data.div_(H_divider)
                    if H_alpha != 1:
                        H.data.add_(H_alpha - 1).relu_()
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
    r"""Probabilistic Latent Component Analysis (PLCA).

    Estimate two marginals :math:`P(c|z)` and :math:`P(n|z)`, which is the matrix W and H, 
    and a prior :math:`P(z)` which is the vector Z, that approximate the observed :math:`P(n,c)`,
    where :math:`P(n,c)` is obtained via ``V / V.sum()`` so the total probabilities sum to 1.

    More precisely:

    .. math::
        P(n, c) \approx \sum_{z}P(c|z)P(z)P(n|z)

    In matrix form:

    .. math::
        V \approx H diag(Z) W^T

    Its formulation is very similar to NMF, but introduce an extra latent vector to incorporate probabilities concept.

    Note:
        If `Vshape` argument is given, the model will try to infer the size of :meth:`W <torchnmf.plca.BaseComponent.W>`,
        :meth:`H <torchnmf.plca.BaseComponent.H>` and :meth:`Z <torchnmf.plca.BaseComponent.Z>`, and override arguments passed through to :meth:`BaseComponent <torchnmf.plca.BaseComponent>`.

    Args:
        Vshape (tuple, optional): size of target matrix V
        rank (int, optional): size of hidden dimension
        **kwargs: arguments passed through to :meth:`BaseComponent <torchnmf.plca.BaseComponent>`


    Shape:
        - V: :math:`(N, C)`
        - W: :math:`(C, R)`
        - H: :math:`(N, R)`
        - Z: :math:`(R,)`


    Examples::

        >>> V = torch.rand(20, 30)
        >>> m = PLCA(V.shape, 5)
        >>> m.W.size()
        torch.Size([30, 5])
        >>> m.H.size()
        torch.Size([20, 5])
        >>> m.Z.size()
        torch.Size([5])
        >>> HZWt = m()
        >>> HZWt.size()
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
    def reconstruct(H, W, Z):
        return torch.einsum("bi,ji,i->bj", H, W, Z)


class SIPLCA(BaseComponent):
    r"""Shift Invariant Probabilistic Latent Component Analysis (SI-PLCA).

    Estimate two marginals :math:`P(c,t|z)` and :math:`P(n,l|z)`, 
    which is the tensor W and H, and a prior :math:`P(z)` which is the vector Z, 
    that approximate the observed :math:`P(n,c,l)`, where :math:`P(n,c,l)` is obtained via ``V / V.sum()`` 
    so the total probabilities sum to 1.

    More precisely:

    .. math::
        P(n, c, l) \approx \sum_{z} \sum_{t} P(c,t|z)P(z)P(n,l-t|z)

    Look at the paper: `Shift-Invariant Probabilistic Latent Component Analysis`_
    by Paris Smaragdis and Bhiksha Raj (2007) for more details.


    Note:
        If `Vshape` argument is given, the model will try to infer the size of :meth:`W <torchnmf.plca.BaseComponent.W>`,
        :meth:`H <torchnmf.plca.BaseComponent.H>` and :meth:`Z <torchnmf.plca.BaseComponent.Z>`, and override arguments passed through to :meth:`BaseComponent <torchnmf.plca.BaseComponent>`.

    Args:
        Vshape (tuple, optional): size of target matrix V
        rank (int, optional): size of hidden dimension
        T (int, optional): size of the convolving window
        **kwargs: arguments passed through to :meth:`BaseComponent <torchnmf.plca.BaseComponent>`


    Shape:
        - V: :math:`(N, C, L_{out})`
        - W: :math:`(C, R, T)`
        - H: :math:`(N, R, L_{in})` 
        - Z: :math:`(R,)` where

        .. math::
            L_{in} = L_{out} - T + 1

    Examples::

        >>> V = torch.rand(33, 50).unsqueeze(0)
        >>> m = SIPLCA(V.shape, 16, 3)
        >>> m.W.size()
        torch.Size([33, 16, 3])
        >>> m.H.size()
        torch.Size([1, 16, 48])
        >>> m.Z.size()
        torch.Size([16])
        >>> HZWt = m()
        >>> HZWt.size()
        torch.Size([1, 33, 50])

    .. _Shift-Invariant Probabilistic Latent Component Analysis:
        https://paris.cs.illinois.edu/pubs/plca-report.pdf

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
    def reconstruct(H, W, Z):
        pad_size = W.shape[2] - 1
        return F.conv1d(H, W.flip(2) * Z.view(-1, 1), padding=pad_size)


class SIPLCA2(BaseComponent):
    r"""Shift Invariant Probabilistic Latent Component Analysis across 2 dimensions (SI-PLCA 2D).

    Estimate two marginals :math:`P(c,k_1,k_2|z)` and :math:`P(n,l,m|z)`, 
    which is the tensor W and H, and a prior :math:`P(z)` which is the vector Z, 
    that approximate the observed :math:`P(n,c,l,m)`, where :math:`P(n,c,l,m)` is obtained via ``V / V.sum()`` 
    so the total probabilities sum to 1.

    More precisely:

    .. math::
        P(n,c,l,m) \approx \sum_{z} \sum_{k_1} \sum_{k_2} P(c,k_1,k_2|z)P(z)P(n,l-k_1,m-k_2|z)

    Look at the paper: `Shift-Invariant Probabilistic Latent Component Analysis`_
    by Paris Smaragdis and Bhiksha Raj (2007) for more details.

    Note:
        If `Vshape` argument is given, the model will try to infer the size of :meth:`W <torchnmf.plca.BaseComponent.W>`,
        :meth:`H <torchnmf.plca.BaseComponent.H>` and :meth:`Z <torchnmf.plca.BaseComponent.Z>`, and override arguments passed through to :meth:`BaseComponent <torchnmf.plca.BaseComponent>`.

    Args:
        Vshape (tuple, optional): size of target tensor V
        rank (int, optional): size of hidden dimension
        kernel_size (int or tuple, optional): size of the convolving kernel
        **kwargs: arguments passed through to :meth:`BaseComponent <torchnmf.plca.BaseComponent>`

    Shape:
        - V: :math:`(N, C, L_{out}, M_{out})`
        - W: :math:`(C, R, \text{kernel_size}[0], \text{kernel_size}[1])`
        - H: :math:`(N, R, L_{in}, M_{in})` 
        - Z: :math:`(R,)` where

        .. math::
            L_{in} = L_{out} - \text{kernel_size}[0] + 1
        .. math::
            M_{in} = M_{out} - \text{kernel_size}[1] + 1

    Examples::

        >>> V = torch.rand(33, 50).unsqueeze(0).unsqueeze(0)
        >>> m = SIPLCA2(V.shape, 16, 3)
        >>> m.W.size()
        torch.Size([1, 16, 3, 3])
        >>> m.H.size()
        torch.Size([1, 16, 31, 48])
        >>> m.Z.size()
        torch.Size([16])
        >>> HZWt = m()
        >>> HZWt.size()
        torch.Size([1, 1, 33, 50])

    .. _Shift-Invariant Probabilistic Latent Component Analysis:
        https://paris.cs.illinois.edu/pubs/plca-report.pdf
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
    def reconstruct(H, W, Z):
        pad_size = (W.shape[2] - 1, W.shape[3] - 1)
        out = F.conv2d(H, W.flip((2, 3)) * Z.view(-1, 1, 1), padding=pad_size)
        return out


class SIPLCA3(BaseComponent):
    r"""Shift Invariant Probabilistic Latent Component Analysis across 3 dimensions (SI-PLCA 3D).

    Estimate two marginals :math:`P(c,k_1,k_2,k_3|z)` and :math:`P(n,l,m,o|z)`, 
    which is the tensor W and H, and a prior :math:`P(z)` which is the vector Z, 
    that approximate the observed :math:`P(n,c,l,m,o)`, where :math:`P(n,c,l,m,o)` is obtained via ``V / V.sum()`` 
    so the total probabilities sum to 1.

    More precisely:

    .. math::
        P(n,c,l,m,o) \approx \sum_{z} \sum_{k_1} \sum_{k_2} \sum_{k_3} P(c,k_1,k_2,k_3|z)P(z)P(n,l-k_1,m-k_2,o-k_3|z)

    Look at the paper: `Shift-Invariant Probabilistic Latent Component Analysis`_
    by Paris Smaragdis and Bhiksha Raj (2007) for more details.

    Note:
        If `Vshape` argument is given, the model will try to infer the size of :meth:`W <torchnmf.plca.BaseComponent.W>`,
        :meth:`H <torchnmf.plca.BaseComponent.H>` and :meth:`Z <torchnmf.plca.BaseComponent.Z>`, and override arguments passed through to :meth:`BaseComponent <torchnmf.plca.BaseComponent>`.

    Args:
        Vshape (tuple, optional): size of target tensor V
        rank (int, optional): size of hidden dimension
        kernel_size (int or tuple, optional): size of the convolving kernel
        **kwargs: arguments passed through to :meth:`BaseComponent <torchnmf.plca.BaseComponent>`


    Shape:
        - V: :math:`(N, C, L_{out}, M_{out}, O_{out})`
        - W: :math:`(C, R, \text{kernel_size}[0], \text{kernel_size}[1], \text{kernel_size}[2])`
        - H: :math:`(N, R, L_{in}, M_{in}, O_{in})`
        - Z: :math:`(R,)` where

        .. math::
            L_{in} = L_{out} - \text{kernel_size}[0] + 1
        .. math::
            M_{in} = M_{out} - \text{kernel_size}[1] + 1
        .. math::
            O_{in} = O_{out} - \text{kernel_size}[2] + 1

    Examples::

        >>> V = torch.rand(3, 64, 64, 100).unsqueeze(0)
        >>> m = SIPLCA3(V.shape, 8, (5, 5, 20))
        >>> m.W.size()
        torch.Size([3, 8, 5, 5, 20])
        >>> m.H.size()
        torch.Size([1, 8, 60, 60, 81])
        >>> m.Z.size()
        torch.Size([8])
        >>> HZWt = m()
        >>> HZWt.size()
        torch.Size([1, 3, 64, 64, 100])

    .. _Shift-Invariant Probabilistic Latent Component Analysis:
        https://paris.cs.illinois.edu/pubs/plca-report.pdf
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
    def reconstruct(H, W, Z):
        pad_size = (W.shape[2] - 1, W.shape[3] - 1, W.shape[4] - 1)
        out = F.conv3d(H, W.flip((2, 3, 4)) *
                       Z.view(-1, 1, 1, 1), padding=pad_size)
        return out
