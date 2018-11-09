from torch import nn
from math import sqrt
from .metrics import Beta_divergence
from .utils import *
from time import time


def _beta_loss_to_float(beta_loss):
    allowed_beta_loss = {'frobenius': 2,
                         'kullback-leibler': 1,
                         'itakura-saito': 0}
    if isinstance(beta_loss, str) and beta_loss in allowed_beta_loss:
        beta_loss = allowed_beta_loss[beta_loss]

    return beta_loss


class NMF(nn.Module):
    """
    Basic NMF model with beta- divergence as loss function.

    Attributes

    H : Tensor, [n_components, n_samples]
        Activation matrix.

    W : Tensor, [n_features, n_compoments]
        Template matrix.

    """

    def __init__(self, Xshape, n_components=None, init_W=None, init_H=None, beta_loss='frobenius', tol=1e-4,
                 max_iter=200, verbose=0, update_W=True, update_H=True, initial_mean=1):
        """

        :param Xshape: Target matrix size.
        :param n_components:
        :param init_W:
        :param init_H:
        :param beta_loss:
        :param tol:
        :param max_iter:
        :param verbose:
        :param update_W:
        :param update_H:
        """
        super().__init__()
        self.K, self.M = Xshape
        self.beta_loss = _beta_loss_to_float(beta_loss)
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.initial_mean = initial_mean
        if not n_components:
            self.n_components = self.K
        else:
            self.n_components = n_components

        avg = sqrt(initial_mean / self.n_components)

        if init_W is None:
            init_W = torch.randn(self.K, self.n_components).mul_(avg).abs_()
            update_W = True
        else:
            init_W = torch.Tensor(init_W)
        self.W = torch.nn.Parameter(init_W, requires_grad=update_W)

        if init_H is None:
            init_H = torch.randn(self.n_components, self.M).mul_(avg).abs_()
            update_H = True
        else:
            init_H = torch.Tensor(init_H)
        self.H = torch.nn.Parameter(init_H, requires_grad=update_H)

    def reset_weights(self, mean=1):
        avg = sqrt(mean / self.n_components)
        for param in self.parameters():
            if param.requires_grad:
                param.data.normal_().mul_(avg).abs_()

    def forward(self, H):
        return self.W @ H

    def _get_W_positive(self, WH, H_sum=None):
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

    def _get_H_positive(self, WH, W_sum=None):
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

    def _loss_fn(self, predict, target):
        return Beta_divergence(predict, target, self.beta_loss)

    def _caculate_loss(self, X, backprop=True):
        self.zero_grad()
        V = self.forward(self.H)
        loss = self._loss_fn(V, X)
        if backprop:
            loss.backward()
        return V, loss

    def fit(self, X):
        start_time = time()

        V, loss = self._caculate_loss(X, False)
        error = self._2sqrt_error(loss)
        previous_error = error_at_init = error

        H_sum, W_sum = None, None
        for n_iter in range(1, self.max_iter + 1):
            if self.W.requires_grad:
                V, loss = self._caculate_loss(X)
                positive_comps, H_sum = self._get_W_positive(V.data, H_sum)
                self._mu_update(self.W, positive_comps)
                W_sum = None

            if self.H.requires_grad:
                V, loss = self._caculate_loss(X)
                positive_comps, W_sum = self._get_H_positive(V.data, W_sum)
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

    def __init__(self, Xshape, T=1, **kwargs):
        """

        :param Xshape: Target matrix size.
        :param T: Size of template.
        :param kwargs: Other arguments that pass to NMF.
        """
        super().__init__(Xshape, **kwargs)
        self.T = T
        avg = sqrt(self.initial_mean / self.n_components / T)
        if self.W.requires_grad and len(self.W.shape) == 2:
            init_W = torch.randn(self.K, self.n_components, T).mul_(avg).abs_()
            self.W = torch.nn.Parameter(init_W)

    def forward(self, H):
        H = F.pad(H, (self.T - 1, 0))
        return F.conv1d(H[None, :], self.W.flip(2))[0]

    def random_weight(self, mean):
        avg = sqrt(mean / self.n_components / self.T)
        for param in self.parameters():
            if param.requires_grad:
                param.data.normal_(avg, avg / 2).abs_()

    def _get_W_positive(self, WH, H_sum=None):
        H = self.H.data
        if self.beta_loss == 1:
            if H_sum is None:
                H_sum = H.sum(1)
            denominator = H_sum[None, :, None]
        else:
            if self.beta_loss != 2:
                WH = WH ** (self.beta_loss - 1)
            H = F.pad(H, (self.T - 1, 0))
            WHHt = F.conv1d(H[:, None], WH[:, None])
            denominator = WHHt.transpose(0, 1).flip(2)

        return denominator, H_sum

    def _get_H_positive(self, WH, W_sum=None):
        W = self.W.data
        if self.beta_loss == 1:
            if W_sum is None:
                W_sum = W.sum((0, 2))
            denominator = W_sum[:, None]
        else:
            if self.beta_loss != 2:
                WH = WH ** (self.beta_loss - 1)
            WH = F.pad(WH, (0, self.T - 1))
            WtWH = F.conv1d(WH[None, :], W.transpose(0, 1))[0]
            denominator = WtWH
        return denominator, W_sum

    def sort(self):
        _, maxidx = self.W.data.sum(2).max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        self.H.data = self.H.data[idx]
