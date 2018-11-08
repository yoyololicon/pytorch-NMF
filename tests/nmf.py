from operator import mul
from functools import reduce
import time

import torch
import torch.nn.functional as F

eps = 1.19209e-07


def _beta_divergence(X, W, H, beta):
    """Compute the beta-divergence of X and dot(W, H).
    Parameters
    ----------
    X : float or array-like, shape (n_samples, n_features)
    W : float or dense array-like, shape (n_samples, n_components)
    H : float or dense array-like, shape (n_components, n_features)
    beta : float, string in {'frobenius', 'kullback-leibler', 'itakura-saito'}
        Parameter of the beta-divergence.
        If beta == 2, this is half the Frobenius *squared* norm.
        If beta == 1, this is the generalized Kullback-Leibler divergence.
        If beta == 0, this is the Itakura-Saito divergence.
        Else, this is the general beta-divergence.
    square_root : boolean, default False
        If True, return np.sqrt(2 * res)
        For beta == 2, it corresponds to the Frobenius norm.
    Returns
    -------
        res : float
            Beta divergence of X and np.dot(X, H)
    """

    # Frobenius norm
    if beta == 2:
        # Avoid the creation of the dense np.dot(W, H) if X is sparse.
        if type(X) == torch.sparse.FloatTensor:
            # norm_X = np.dot(X.data, X.data)
            # norm_WH = trace_dot(np.dot(np.dot(W.T, W), H), H)
            # cross_prod = trace_dot((X * H.T), W)
            # res = (norm_X + norm_WH - 2. * cross_prod) / 2.
            raise NotImplementedError("the sparse method will be implement in future")
        else:
            res = F.mse_loss(W @ H, X, reduction='sum') / 2
            return res

    if type(X) == torch.sparse.FloatTensor:
        # compute np.dot(W, H) only where X is nonzero
        # WH_data = _special_sparse_dot(W, H, X).data
        # X_data = X.data
        raise NotImplementedError("the sparse method will be implement in future")
    else:
        WH = W @ H
        WH_data = WH.view(-1)
        X_data = X.view(-1)

    # generalized Kullback-Leibler divergence
    if beta == 1:
        # fast and memory efficient computation of np.sum(np.dot(W, H))
        sum_WH = W.sum(0) @ H.sum(1)
        # computes np.sum(X * log(X / WH)) only where X is nonzero
        div = X_data / WH_data
        res = X_data @ torch.log(div + eps) + sum_WH - X_data.sum()

    # Itakura-Saito divergence
    elif beta == 0:
        div = X_data / WH_data
        res = div.sum() - reduce(mul, X.shape) - torch.log(div + eps).sum()

    # beta-divergence, beta not in (0, 1, 2)
    else:
        if type(X) == torch.sparse.FloatTensor:
            # slow loop, but memory efficient computation of :
            # np.sum(np.dot(W, H) ** beta)
            # sum_WH_beta = 0
            # for i in range(X.shape[1]):
            #    sum_WH_beta += np.sum(np.dot(W, H[:, i]) ** beta)
            raise NotImplementedError("the sparse method will be implement in future")
        else:
            sum_WH_beta = torch.sum(WH ** beta)

        sum_X_WH = X_data @ WH_data ** (beta - 1)
        res = ((X_data ** beta).sum() - beta * sum_X_WH + sum_WH_beta * (beta - 1)) / (beta * (beta - 1))

    return res


def _beta_loss_to_float(beta_loss):
    """Convert string beta_loss to float"""
    allowed_beta_loss = {'frobenius': 2,
                         'kullback-leibler': 1,
                         'itakura-saito': 0}
    if isinstance(beta_loss, str) and beta_loss in allowed_beta_loss:
        beta_loss = allowed_beta_loss[beta_loss]

    return beta_loss


def _multiplicative_update_w(X, W, H, beta_loss, H_sum=None, HHt=None):
    """update W in Multiplicative Update NMF"""
    # only denominator is computed
    if beta_loss == 2:
        if HHt is None:
            HHt = H @ H.t()
        denominator = W @ HHt
    elif beta_loss == 1:
        if H_sum is None:
            H_sum = H.sum(1)  # shape(n_components, )
        denominator = H_sum[None, :]
    else:
        WH = W @ H

        # computation of WHHt = dot(dot(W, H) ** beta_loss - 1, H.T)
        if type(X) == torch.sparse.FloatTensor:
            # memory efficient computation
            # (compute row by row, avoiding the dense matrix WH)
            # WHHt = np.empty(W.shape)
            # for i in range(X.shape[0]):
            #    WHi = np.dot(W[i, :], H)
            #    if beta_loss - 1 < 0:
            #        WHi[WHi == 0] = EPSILON
            #    WHi **= beta_loss - 1
            #    WHHt[i, :] = np.dot(WHi, H.T)
            raise NotImplementedError("the sparse method will be implement in future")
        else:
            WH = WH ** (beta_loss - 1)
            WHHt = WH @ H.t()
        denominator = WHHt

    return denominator, H_sum, HHt


def _multiplicative_update_h(X, W, H, beta_loss, W_sum=None, WtW=None):
    """update H in Multiplicative Update NMF"""
    if beta_loss == 2:
        if WtW is None:
            WtW = W.t() @ W
        denominator = WtW @ H

    elif beta_loss == 1:
        if W_sum is None:
            W_sum = W.sum(0)  # shape(n_components, )
        denominator = W_sum[:, None]
    else:
        WH = W @ H
        # computation of WtWH = dot(W.T, dot(W, H) ** beta_loss - 1)
        if type(X) == torch.sparse.FloatTensor:
            # memory efficient computation
            # (compute column by column, avoiding the dense matrix WH)
            # WtWH = np.empty(H.shape)
            # for i in range(X.shape[1]):
            #    WHi = np.dot(W, H[:, i])
            #    if beta_loss - 1 < 0:
            #        WHi[WHi == 0] = EPSILON
            #    WHi **= beta_loss - 1
            #    WtWH[:, i] = np.dot(W.T, WHi)
            raise NotImplementedError("the sparse method will be implement in future")
        else:
            WH = WH ** (beta_loss - 1)
            WtWH = W.t() @ WH
        denominator = WtWH

    return denominator, W_sum, WtW


class NMF(torch.nn.Module):
    def __init__(self, X, W=None, H=None, n_components=None, beta_loss='frobenius', tol=1e-4, max_iter=200, verbose=0):
        super().__init__()
        self.n_components = n_components
        self.beta_loss = _beta_loss_to_float(beta_loss)
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose

        self.X = torch.nn.Parameter(torch.Tensor(X), requires_grad=False)
        self.K, self.M = self.X.shape
        if not n_components:
            self.n_components = self.K

        require_grad = True
        if W is None:
            init_W = torch.Tensor(self.K, self.n_components)
        else:
            init_W = torch.Tensor(W)
            require_grad = False
        self.W = torch.nn.Parameter(init_W, requires_grad=require_grad)

        require_grad = True
        if H is None:
            init_H = torch.Tensor(self.n_components, self.M)
        else:
            init_H = torch.Tensor(H)
            require_grad = False
        self.H = torch.nn.Parameter(init_H, requires_grad=require_grad)

        self.reset_weight()

    def reset_weight(self):
        """
        Initialize the value of weight equally sample from 0~1 when it's not fixed.

        :return: None.
        """
        avg = torch.sqrt(self.X.mean() / self.n_components)
        for param in self.parameters():
            if param.requires_grad:
                param.data.normal_(avg).abs_()

    def fit(self):
        return self._fit_multiplicative_update()

    def forward(self, H):
        return self.W @ H

    def _fit_multiplicative_update(self):
        # X, W, H are old nn.Parameter
        start_time = time.time()

        # used for the convergence criterion
        error = (_beta_divergence(self.X, self.W, self.H, self.beta_loss) * 2).sqrt().item()

        previous_error = error_at_init = error

        H_sum, HHt, W_sum, WtW = None, None, None, None
        for n_iter in range(1, self.max_iter + 1):
            # update W
            if self.W.requires_grad:
                self.zero_grad()
                loss = _beta_divergence(self.X, self.W, self.H, self.beta_loss)
                loss.backward()
                # H_sum, HHt and XHt are saved and reused if not update_H
                positive_comps, H_sum, HHt = _multiplicative_update_w(self.X.data, self.W.data, self.H.data,
                                                                      self.beta_loss, H_sum, HHt)
                neg = positive_comps - self.W.grad.data
                self.W.data.mul_(neg / positive_comps)
                # self.W.data.add_(-self.W.data * self.W.grad.data / positive_comps)

                W_sum, WtW = None, None

            # update H
            if self.H.requires_grad:
                self.zero_grad()
                loss = _beta_divergence(self.X, self.W, self.H, self.beta_loss)
                loss.backward()
                positive_comps, W_sum, WtW = _multiplicative_update_h(self.X, self.W, self.H, self.beta_loss, W_sum,
                                                                      WtW)
                neg = positive_comps - self.H.grad.data
                self.H.data.mul_(neg / positive_comps)
                # self.H.data.add_(- self.H.data * self.H.grad.data / positive_comps)

                # These values will be recomputed since H changed
                H_sum, HHt = None, None

            # test convergence criterion every 10 iterations
            if self.tol > 0 and n_iter % 10 == 0:
                error = (loss * 2).sqrt().item()

                if self.verbose:
                    iter_time = time.time()
                    print("Epoch %02d reached after %.3f seconds, error: %f" % (n_iter, iter_time - start_time, error))

                if (previous_error - error) / error_at_init < self.tol:
                    break
                previous_error = error

        # do not print if we have already printed in the convergence test
        if self.verbose and (self.tol == 0 or n_iter % 10 != 0):
            end_time = time.time()
            print("Epoch %02d reached after %.3f seconds." % (n_iter, end_time - start_time))

        return n_iter


if __name__ == '__main__':
    import librosa
    from librosa import display
    import numpy as np
    import matplotlib.pyplot as plt

    # torch.set_flush_denormal(True)
    # torch.set_default_tensor_type(torch.DoubleTensor)

    y, sr = librosa.load(
        '/media/ycy/Shared/Datasets/DSD100subset/Sources/Test/005 - Angela Thomas Wade - Milk Cow Blues/drums.wav')
    y = torch.from_numpy(y)
    windowsize = 2048
    S = torch.stft(y, windowsize, window=torch.hann_window(windowsize)).pow(2).sum(2).sqrt()
    R = 4
    max_iter = 1000

    net = NMF(S, n_components=R, max_iter=max_iter, verbose=True, beta_loss=2)
    start = time.time()
    n_iter = net.fit()
    print(n_iter / (time.time() - start))

    W = net.W
    H = net.H
    V = W @ H

    plt.subplot(3, 1, 1)
    display.specshow(librosa.amplitude_to_db(W.detach().cpu().numpy(), ref=np.max), y_axis='log')
    plt.title('Template ')
    plt.subplot(3, 1, 2)
    display.specshow(H.detach().cpu().numpy(), x_axis='time')
    plt.colorbar()
    plt.title('Activations')
    plt.subplot(3, 1, 3)
    display.specshow(librosa.amplitude_to_db(V.detach().cpu().numpy(), ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed spectrogram')
    plt.tight_layout()
    plt.show()
