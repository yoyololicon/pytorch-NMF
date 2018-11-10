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

    def __init__(self, Xshape, n_components=None):
        """

        :param Xshape: Target matrix size.
        :param n_components:
        """
        super().__init__()
        self.K, self.M = Xshape
        if not n_components:
            self.n_components = self.K
        else:
            self.n_components = n_components

        self.W = torch.nn.Parameter(torch.rand(self.K, self.n_components))
        self.H = torch.nn.Parameter(torch.rand(self.n_components, self.M))

    def _random_weight(self, param, mean):
        avg = torch.sqrt(mean / self.n_components)
        param.data.normal_().mul_(avg).abs_()

    def forward(self, H):
        return self.W @ H

    def _get_W_positive(self, WH, beta, H_sum=None):
        H = self.H.data
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
        W = self.W.data
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
        multiplier = pos - param.grad.data
        if l1_reg > 0:
            pos.add_(l1_reg)
        if l2_reg > 0:
            if pos.shape != param.data.shape:
                pos = pos + l2_reg * param.data
            else:
                pos.add_(l2_reg * param.data)
        multiplier.div_(pos)
        if gamma != 1:
            multiplier.pow_(gamma)
        param.data.mul_(multiplier)

    def _2sqrt_error(self, x):
        return sqrt(x * 2)

    def _caculate_loss(self, X, beta, backprop=True):
        self.zero_grad()
        V = self.forward(self.H)
        loss = Beta_divergence(V, X, beta)
        if backprop:
            loss.backward()
        return V, loss

    def fit(self,
            X,
            W=None,
            H=None,
            update_W=True,
            update_H=True,
            beta_loss='frobenius',
            tol=1e-4,
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

        V, loss = self._caculate_loss(X, beta, False)
        error = self._2sqrt_error(loss.item())
        previous_error = error_at_init = error

        start_time = time()
        H_sum, W_sum = None, None
        for n_iter in range(1, max_iter + 1):
            if self.W.requires_grad:
                V, loss = self._caculate_loss(X, beta)
                positive_comps, H_sum = self._get_W_positive(V.data, beta, H_sum)
                self._mu_update(self.W, positive_comps, gamma, l1_reg, l2_reg)
                W_sum = None

            if self.H.requires_grad:
                V, loss = self._caculate_loss(X, beta)
                positive_comps, W_sum = self._get_H_positive(V.data, beta, W_sum)
                self._mu_update(self.H, positive_comps, gamma, l1_reg, l2_reg)
                H_sum = None

            # test convergence criterion every 10 iterations
            if tol > 0 and n_iter % 10 == 0:
                error = self._2sqrt_error(loss.item())
                if verbose:
                    iter_time = time()
                    print("Epoch %02d reached after %.3f seconds, error: %f" % (n_iter, iter_time - start_time, error))

                if (previous_error - error) / error_at_init < tol:
                    break
                previous_error = error

        # do not print if we have already printed in the convergence test
        if verbose and (tol == 0 or n_iter % 10 != 0):
            end_time = time()
            print("Epoch %02d reached after %.3f seconds." % (n_iter, end_time - start_time))

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
        self.W = torch.nn.Parameter(torch.rand(self.K, self.n_components, self.T))

    def forward(self, H):
        H = F.pad(H, (self.T - 1, 0))
        return F.conv1d(H[None, :], self.W.flip(2))[0]

    def _get_W_positive(self, WH, beta, H_sum=None):
        H = self.H.data
        if beta == 1:
            if H_sum is None:
                H_sum = H.sum(1)
            denominator = H_sum[None, :, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            H = F.pad(H, (self.T - 1, 0))
            WHHt = F.conv1d(H[:, None], WH[:, None])
            denominator = WHHt.transpose(0, 1).flip(2)

        return denominator, H_sum

    def _get_H_positive(self, WH, beta, W_sum=None):
        W = self.W.data
        if beta == 1:
            if W_sum is None:
                W_sum = W.sum((0, 2))
            denominator = W_sum[:, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WH = F.pad(WH, (0, self.T - 1))
            WtWH = F.conv1d(WH[None, :], W.transpose(0, 1))[0]
            denominator = WtWH
        return denominator, W_sum

    def sort(self):
        _, maxidx = self.W.data.sum(2).max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        self.H.data = self.H.data[idx]
