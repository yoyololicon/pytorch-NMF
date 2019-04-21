import torch
import torch.nn.functional as F
from torch.nn import Parameter
from math import sqrt
from .metrics import Beta_divergence
from .base import Base
from tqdm import tqdm

import numpy as np


def _mu_update(param, pos, gamma, l1_reg, l2_reg):
    if param.grad is None:
        return
    multiplier = pos - param.grad
    if l1_reg > 0:
        pos.add_(l1_reg)
    if l2_reg > 0:
        if pos.shape != param.data.shape:
            pos = pos + l2_reg * param
        else:
            pos.add_(l2_reg * param)
    multiplier.div_(pos)
    if gamma != 1:
        multiplier.pow_(gamma)
    param.mul_(multiplier)


class NMF_(Base):
    def __init__(self, W_size, H_size, rank):
        super().__init__()
        self.rank = rank
        self.W = Parameter(torch.rand(*W_size))
        self.H = Parameter(torch.rand(*H_size))

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

    def fit(self,
            V,
            W=None,
            H=None,
            update_W=True,
            update_H=True,
            beta=1,
            tol=1e-5,
            max_iter=200,
            verbose=0,
            initial='random',
            alpha=0,
            l1_ratio=0
            ):
        if W is None:
            pass  # will do special initialization in thre future
        else:
            self.W.data.copy_(W)
            self.W.requires_grad = update_W

        if H is None:
            pass
        else:
            self.H.data.copy_(H)
            self.H.requires_grad = update_H

        if beta < 1:
            gamma = 1 / (2 - beta)
        elif beta > 2:
            gamma = 1 / (beta - 1)
        else:
            gamma = 1

        l1_reg = alpha * l1_ratio
        l2_reg = alpha * (1 - l1_ratio)

        WH = self.forward()
        loss_scale = torch.prod(torch.tensor(V.shape)).float()
        loss = Beta_divergence(WH, V, beta) / loss_scale
        previous_loss = loss_init = loss

        H_sum, W_sum = None, None
        with tqdm(total=max_iter) as pbar:
            for n_iter in range(1, max_iter + 1):
                if self.W.requires_grad:
                    self.zero_grad()
                    WH = self.reconstruct(self.H.detach(), self.W)
                    loss = Beta_divergence(V, WH, beta)
                    loss.backward()

                    with torch.no_grad():
                        positive_comps, H_sum = self.get_W_positive(WH, beta, H_sum)
                        _mu_update(self.W, positive_comps, gamma, l1_reg, l2_reg)
                    W_sum = None

                if self.H.requires_grad:
                    self.zero_grad()
                    WH = self.reconstruct(self.H, self.W.detach())
                    loss = Beta_divergence(WH, V, beta)
                    loss.backward()

                    with torch.no_grad():
                        positive_comps, W_sum = self.get_H_positive(V, beta, W_sum)
                        _mu_update(self.H, positive_comps, gamma, l1_reg, l2_reg)
                    H_sum = None

                loss = loss.item() / loss_scale
                if verbose:
                    pbar.set_postfix(loss=loss)
                    # pbar.set_description('Beta loss=%.4f' % error)
                    pbar.update()

                if (previous_loss - loss) / loss_init < tol:
                    break
                previous_loss = loss

        return n_iter

    def fit_transform(self, *args, **kwargs):
        n_iter = self.fit(*args, **kwargs)
        return n_iter, self.forward()


class NMF(NMF_):

    def __init__(self, Xshape, rank=None):
        self.K, self.M = Xshape
        if not rank:
            rank = self.K

        super().__init__((self.K, rank), (rank, self.M), rank)

    def reconstruct(self, H, W):
        return W @ H

    def get_W_positive(self, WH, beta, H_sum):
        H = self.H
        if beta == 1:
            if H_sum is None:
                H_sum = H.sum(1)
            denominator = H_sum[None, :]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WHHt = WH @ H.t()
            denominator = WHHt

        return denominator, H_sum

    def get_H_positive(self, WH, beta, W_sum):
        W = self.W
        if beta == 1:
            if W_sum is None:
                W_sum = W.sum(0)  # shape(n_components, )
            denominator = W_sum[:, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WtWH = W.t() @ WH
            denominator = WtWH
        return denominator, W_sum

    def sort(self):
        _, maxidx = self.W.data.max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        self.H.data = self.H.data[idx]


class NMFD(NMF_):
    def __init__(self, Xshape, T=1, rank=None):
        self.K, self.M = Xshape
        if not rank:
            rank = self.K
        self.pad_size = T - 1
        super().__init__((self.K, rank, T), (rank, self.M - T + 1), rank)

    def reconstruct(self, H, W):
        return F.conv1d(H[None, :], W.flip(2), padding=self.pad_size)[0]

    def get_W_positive(self, WH, beta, H_sum):
        H = self.H
        if beta == 1:
            if H_sum is None:
                H_sum = H.sum(1)
            denominator = H_sum[None, :, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WHHt = F.conv1d(WH[:, None], H[:, None])
            denominator = WHHt

        return denominator, H_sum

    def _get_H_positive(self, WH, beta, W_sum):
        W = self.W
        if beta == 1:
            if W_sum is None:
                W_sum = W.sum((0, 2))
            denominator = W_sum[:, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WtWH = F.conv1d(WH[None, :], W.transpose(0, 1))[0]
            denominator = WtWH
        return denominator, W_sum

    def sort(self):
        _, maxidx = self.W.data.sum(2).max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        self.H.data = self.H.data[idx]
