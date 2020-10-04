import torch
from torch.nn import Parameter
import torch.nn.functional as F
from typing import Union, Iterable
from collections.abc import Iterable as Iterabc
from .base import Base
from .metrics import Beta_divergence
from tqdm import tqdm


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
    def __init__(self,
                 rank=None,
                 batch=1,
                 W: Union[Iterable[int], torch.Tensor] = None,
                 H: Union[Iterable[int], torch.Tensor] = None,
                 trainable_W=True,
                 trainable_H=True,
                 W_constraints={},
                 H_constraints={}):
        super().__init__()

        if isinstance(W, Iterabc) and trainable_W:
            self.register_parameter('W', Parameter(torch.rand(*W)))
        elif isinstance(W, torch.Tensor):
            self.register_parameter('W', Parameter(torch.empty(*W.size()), requires_grad=trainable_W))
            self.W.data.copy_(W)
        else:
            self.register_parameter('W', None)

        if isinstance(H, Iterabc) and trainable_H:
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
    def get_W_kl_positive(H) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def get_H_kl_positive(W) -> torch.Tensor:
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

    def fit_transform(self, *args, sparse=False, **kwargs):
        n_iter = self.fit(*args, **kwargs)
        return n_iter, self.forward()


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
    def get_W_kl_positive(H):
        return H.sum(1)

    @staticmethod
    def get_H_kl_positive(W):
        W_sum = W.sum(0)
        return W_sum[:, None]


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
    def get_W_kl_positive(H) -> torch.Tensor:
        if H.dim() < 3:
            H = H.unsqueeze(0)
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
    def get_W_kl_positive(H) -> torch.Tensor:
        if H.dim() < 4:
            H = H.unsqueeze(0)
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
    def get_W_kl_positive(H) -> torch.Tensor:
        if H.dim() < 5:
            H = H.unsqueeze(0)
        return H.sum((0, 2, 3, 4), keepdims=True)

    @staticmethod
    def get_H_kl_positive(W) -> torch.Tensor:
        W_sum = W.sum((0, 2, 3, 4))
        return W_sum[:, None, None, None]
