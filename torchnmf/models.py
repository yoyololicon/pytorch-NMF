from torch import nn
from .metrics import *

import collections
from time import time


def _beta_loss_to_float(beta_loss):
    allowed_beta_loss = {'frobenius': 2,
                         'kullback-leibler': 1,
                         'itakura-saito': 0}
    if isinstance(beta_loss, str) and beta_loss in allowed_beta_loss:
        beta_loss = allowed_beta_loss[beta_loss]

    return beta_loss


class NMF(nn.Module):

    def __init__(self, Xshape, n_components=None, init_W=None, init_H=None, beta_loss='frobenius', tol=1e-4,
                 max_iter=200, verbose=0, update_W=True, update_H=True):
        super().__init__()
        self.K, self.M = Xshape
        self.beta_loss = _beta_loss_to_float(beta_loss)
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        if not n_components:
            self.n_components = self.K
        else:
            self.n_components = n_components

        if init_W is None:
            init_W = torch.Tensor(self.K, self.n_components)
            update_W = True
        else:
            init_W = torch.Tensor(init_W)
        self.W = torch.nn.Parameter(init_W, requires_grad=update_W)

        if init_H is None:
            init_H = torch.Tensor(self.n_components, self.M)
            update_H = True
        else:
            init_H = torch.Tensor(init_H)
        self.H = torch.nn.Parameter(init_H, requires_grad=update_H)

    def random_weight(self, mean):
        avg = torch.sqrt(mean / self.n_components)
        for param in self.parameters():
            if param.requires_grad:
                param.data.normal_(avg, avg / 2).abs_()

    def forward(self, H):
        return self.W @ H

    def get_W_positive(self, WH, H_sum=None):
        H = self.H.data
        if self.beta_loss == 1:
            if H_sum is None:
                H_sum = H.sum(1)  # shape(n_components, )
            denominator = H_sum[None, :]
        else:
            if self.beta_loss != 2:
                WH = WH ** (self.beta_loss - 1)
            WHHt = WH @ H.t()
            denominator = WHHt

        return denominator, H_sum

    def get_H_positive(self, WH, W_sum=None):
        W = self.W.data
        if self.beta_loss == 1:
            if W_sum is None:
                W_sum = W.sum(0)  # shape(n_components, )
            denominator = W_sum[:, None]
        else:
            if self.beta_loss != 2:
                WH = WH ** (self.beta_loss - 1)
            WtWH = W.t() @ WH
            denominator = WtWH
        return denominator, W_sum

    def _mu_update(self, param, pos):
        if param.grad is None:
            return
        neg = pos - param.grad.data
        param.data.mul_(neg / pos)

    def _2sqrt_error(self, x):
        return (x * 2).sqrt().item()

    def loss_fn(self, predict, target):
        return Beta_divergence(predict, target, self.beta_loss)

    def _caculate_loss(self, X, backprop=True):
        self.zero_grad()
        V = self.forward(self.H)
        loss = self.loss_fn(V, X)
        if backprop:
            loss.backward()
        return V, loss

    def fit(self, X):
        self.random_weight(X.mean())
        start_time = time()

        V, loss = self._caculate_loss(X, False)
        error = self._2sqrt_error(loss)
        previous_error = error_at_init = error

        H_sum, W_sum = None, None
        for n_iter in range(1, self.max_iter + 1):
            if self.W.requires_grad:
                V, loss = self._caculate_loss(X)
                positive_comps, H_sum = self.get_W_positive(V.data, H_sum)
                self._mu_update(self.W, positive_comps)
                W_sum = None

            if self.H.requires_grad:
                V, loss = self._caculate_loss(X)
                positive_comps, W_sum = self.get_H_positive(V.data, W_sum)
                self._mu_update(self.H, positive_comps)
                H_sum = None

            # test convergence criterion every 10 iterations
            if self.tol > 0 and n_iter % 10 == 0:
                error = self._2sqrt_error(loss)
                if self.verbose:
                    iter_time = time()
                    print("Epoch %02d reached after %.3f seconds, error: %f" % (n_iter, iter_time - start_time, error))

                if (previous_error - error) / error_at_init < self.tol:
                    break
                previous_error = error

        # do not print if we have already printed in the convergence test
        if self.verbose and (self.tol == 0 or n_iter % 10 != 0):
            end_time = time()
            print("Epoch %02d reached after %.3f seconds." % (n_iter, end_time - start_time))

        return n_iter

    def fit_transform(self, X):
        n_iter = self.fit(X)
        return n_iter, self.forward(self.H)


class NMFD(NMF):
    """
    NMF deconvolution model.
    """

    def __init__(self, Xshape, T=1, **kwargs):
        super().__init__(Xshape, **kwargs)
        self.T = T
        if self.W.requires_grad and len(self.W.shape) == 2:
            self.W = torch.nn.Parameter(torch.Tensor(self.K, self.n_components, self.T))

    def forward(self, H):
        return F.conv1d(H[None, :], self.W.flip(2), padding=self.T - 1)[0, :, :H.shape[1]]

    def random_weight(self, mean):
        avg = torch.sqrt(mean / self.n_components / self.T)
        for param in self.parameters():
            if param.requires_grad:
                param.data.normal_(avg, avg / 2).abs_()

    def get_W_positive(self, WH, H_sum=None):
        H = self.H.data
        if self.beta_loss == 1:
            if H_sum is None:
                H_sum = H.sum(1)  # shape(n_components, )
            denominator = H_sum[None, :, None]
        else:
            if self.beta_loss != 2:
                WH = WH ** (self.beta_loss - 1)
            Ht = F.pad(H, pad=(self.T - 1, 0)).unfold(1, self.M, 1).transpose(1, 2)
            WHHt = WH @ Ht
            denominator = WHHt.transpose(0, 1).flip(2)

        return denominator, H_sum

    def get_H_positive(self, WH, W_sum=None):
        W = self.W.data
        if self.beta_loss == 1:
            if W_sum is None:
                W_sum = W.sum((0, 2))  # shape(n_components, )
            denominator = W_sum[:, None]
        else:
            if self.beta_loss != 2:
                WH = WH ** (self.beta_loss - 1)
            WtWH = F.conv1d(WH[None, :], W.transpose(0, 1), padding=self.T - 1)[0, :, -self.M:]
            denominator = WtWH
        return denominator, W_sum
