import torch
import torch.nn.functional as F
from torch import nn
from math import sqrt
from .metrics import Beta_divergence
from tqdm import tqdm

import numpy as np


def _beta_loss_to_float(beta_loss):
    allowed_beta_loss = {'frobenius': 2,
                         'kullback-leibler': 1,
                         'itakura-saito': 0}
    if isinstance(beta_loss, str) and beta_loss in allowed_beta_loss:
        beta_loss = allowed_beta_loss[beta_loss]

    return beta_loss



class NMF(Base):

    def __init__(self, Xshape, rank=None):
        super().__init__()
        self.K, self.M = Xshape
        if not rank:
            self.rank = self.K
        else:
            self.rank = rank

        self.W = torch.nn.Parameter(torch.rand(self.K, self.rank))
        self.H = torch.nn.Parameter(torch.rand(self.rank, self.M))

    def forward(self, H=None, W=None):
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        return W @ H

    def _get_W_positive(self, WH, beta, H_sum=None):
        H = self.H
        if beta == 1:
            if H_sum is None:
                H_sum = H.sum(1)  # shape(n_components, )
            denominator = H_sum[None, :]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WHHt = WH @ H.t()
            denominator = WHHt

        return denominator, H_sum

    def _get_H_positive(self, WH, beta, W_sum=None):
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

    def _mu_update(self, param, pos, gamma, l1_reg, l2_reg):
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

    def _2sqrt_error(self, x):
        return sqrt(x * 2)

    def fit(self,
            X,
            W=None,
            H=None,
            update_W=True,
            update_H=True,
            beta_loss='frobenius',
            tol=1e-5,
            max_iter=200,
            verbose=0,
            initial='random',
            alpha=0,
            l1_ratio=0
            ):
        if W is None:
            if initial == 'random':
                self._random_weight(self.W, X.mean())
        else:
            self.W.data.copy_(W)
            self.W.requires_grad = update_W

        if H is None:
            if initial == 'random':
                self._random_weight(self.H, X.mean())
        else:
            self.H.data.copy_(H)
            self.H.requires_grad = update_H
        beta = _beta_loss_to_float(beta_loss)

        if beta < 1:
            gamma = 1 / (2 - beta)
        elif beta > 2:
            gamma = 1 / (beta - 1)
        else:
            gamma = 1

        l1_reg = alpha * l1_ratio
        l2_reg = alpha * (1 - l1_ratio)

        V = self.forward()
        loss = Beta_divergence(V, X, beta)
        error = self._2sqrt_error(loss.item())
        previous_error = error_at_init = error

        H_sum, W_sum = None, None
        with tqdm(total=max_iter) as pbar:
            for n_iter in range(1, max_iter + 1):
                if self.W.requires_grad:
                    self.zero_grad()
                    V = self.forward(H=self.H.detach())
                    loss = Beta_divergence(V, X, beta)
                    loss.backward()

                    with torch.no_grad():
                        positive_comps, H_sum = self._get_W_positive(V, beta, H_sum)
                        self._mu_update(self.W, positive_comps, gamma, l1_reg, l2_reg)
                    W_sum = None

                if self.H.requires_grad:
                    self.zero_grad()
                    V = self.forward(W=self.W.detach())
                    loss = Beta_divergence(V, X, beta)
                    loss.backward()

                    with torch.no_grad():
                        positive_comps, W_sum = self._get_H_positive(V, beta, W_sum)
                        self._mu_update(self.H, positive_comps, gamma, l1_reg, l2_reg)
                    H_sum = None

                error = self._2sqrt_error(loss.item())
                if verbose:
                    # pbar.set_postfix(loss=error)
                    pbar.set_description('Beta loss=%.4f' % error)
                    pbar.update()

                if (previous_error - error) / error_at_init < tol:
                    break
                previous_error = error

        return n_iter

    def fit_transform(self, *args, **kwargs):
        n_iter = self.fit(*args, **kwargs)
        return n_iter, self.forward(self.H)

    def sort(self):
        _, maxidx = self.W.data.max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        self.H.data = self.H.data[idx]


class NMFD(NMF):
    """
    NMF deconvolution model.

    Attributes

    H : Tensor, [n_components, n_samples]
        Activation matrix.

    W : Tensor, [n_features, n_compoments, n_timesteps]
        Template matrix.

    """

    def __init__(self, Xshape, T=1, n_components=None):
        super().__init__(Xshape, n_components)
        self.T = T
        self.W = torch.nn.Parameter(torch.rand(self.K, self.rank, self.T))
        self.H = torch.nn.Parameter(torch.rand(self.rank, self.M - self.T + 1))

    def forward(self, H=None, W=None):
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        return F.conv1d(H[None, :], W.flip(2), padding=self.T - 1)[0]

    def _get_W_positive(self, WH, beta, H_sum=None):
        H = self.H
        if beta == 1:
            if H_sum is None:
                H_sum = H.sum(1)
            denominator = H_sum[None, :, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WHHt = F.conv1d(H[:, None], WH[:, None], padding=self.T - 1)
            denominator = WHHt.transpose(0, 1).flip(2)

        return denominator, H_sum

    def _get_H_positive(self, WH, beta, W_sum=None):
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



