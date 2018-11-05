from torch import nn
from .utils import *
from .metrics import *

import collections


class _NMF(nn.Module):
    """
    This is the base class for any NMF model.

    .. attribute:: W

        Basis matrix -- the first matrix factor in standard factorization

    .. attribute:: H

        Mixture matrix -- the second matrix factor in standard factorization
    """

    def __init__(self, K, M, R, W, H, fix_W, fix_H):
        """
        Construct generic NMF model.

        :param K: First dimension of the target matrix.
        :param M: Second dimension of the target matrix.
        :param R: Number of components.
        :param W: Can be a tuple specify each dimension of Basis Matrix W,
            or a torch.Tensor or np.ndarray as initial value.
        :param H: Can be a tuple specify each dimension of Mixture Matrix H,
            or a torch.Tensor or np.ndarray as initial value.
        :param fix_W: Whether to fix W or not.
        :param fix_H: Whether to fix H or not.
        """
        super().__init__()
        self.K = K
        self.M = M
        self.R = R
        if isinstance(W, collections.Iterable):
            init_W = torch.Tensor(*W)
        else:
            init_W = return_torch(W)
            self.fix_W = fix_W

        if isinstance(H, collections.Iterable):
            init_H = torch.Tensor(*H)
        else:
            init_H = return_torch(H)
            self.fix_H = fix_H

        self.W = nn.Parameter(init_W, requires_grad=not fix_W)
        self.H = nn.Parameter(init_H, requires_grad=not fix_H)
        self.reset_weight()

    def reset_weight(self):
        """
        Initialize the value of weight equally sample from 0~1 when it's not fixed.

        :return: None.
        """
        for param in self.parameters():
            if param.requires_grad:
                param.data.uniform_()

    def loss_fn(self, predict, target):
        """

        :return: loss
        """
        raise NotImplementedError

    def forward(self, V):
        """
        Fit the model based on target matrix V.

        :param V: Target matrix.
        :param n_iter: Number of iterations.
        :return: W, H.
        """
        V_tilde = self.reconstruct()
        loss = self.loss_fn(V_tilde, V)
        return V_tilde, loss

    def update_W(self):
        """
        Update mechanism for Basis Matrix W.

        :param VV: Element-wise ratio of the target matrix and the reconstructed matrix.
        :return: updated W.
        """
        pos = self.get_W_positive()
        self._mu_update(self.W, pos)
        return self.W

    def update_H(self):
        """
        Update mechanism for Mixture Matrix H.

        :param VV: Element-wise ratio of the target matrix and the reconstructed matrix.
        :return: updated H.
        """
        pos = self.get_H_positive()
        self._mu_update(self.H, pos)
        return self.H

    def get_W_positive(self):
        raise NotImplementedError

    def get_H_positive(self):
        raise NotImplementedError

    def _mu_update(self, param, pos):
        """

        :param param:
        :param pos:
        :return: None
        """
        if param.grad is None:
            return
        neg = pos - param.grad.data
        param.data.mul_(neg / pos)

    def reconstruct(self):
        return self.W @ self.H


class NMF_L2(_NMF):
    """
    Standard NMF model.
    """

    def __init__(self, Vshape, R, W=None, H=None, fix_W=False, fix_H=False):
        """
        :param Vshape: Shape of the target matrix.
        :type Vshape: tuple.
    
        :param R: Number of components.
        :param W: A torch.Tensor or np.ndarray as initial value or None.
        :param H: A torch.Tensor or np.ndarray as initial value or None.
        :param fix_W: Whether to fix W or not.
        :param fix_H: Whether to fix H or not.
        """
        K, M = Vshape
        if W is None:
            W = (K, R)
        if H is None:
            H = (R, M)
        super().__init__(K, M, R, W, H, fix_W, fix_H)

    def get_W_positive(self):
        return self.W.data @ self.H.data @ self.H.data.t()

    def get_H_positive(self):
        return self.W.data.t() @ self.W.data @ self.H.data

    def loss_fn(self, predict, target):
        return Euclidean(predict, target)


class NMF_KL(NMF_L2):
    """
    Standard NMF model.
    """

    def get_W_positive(self):
        return self.H.data.sum(1, keepdim=True).t()

    def get_H_positive(self):
        return self.W.data.sum(0, keepdim=True).t()

    def loss_fn(self, predict, target):
        return KL_divergence(predict, target)


class NMF_IS(NMF_L2):

    def get_W_positive(self):
        return (1 / (self.W.data @ self.H.data)) @ self.H.data.t()

    def get_H_positive(self):
        return self.W.data.t() @ (1 / (self.W.data @ self.H.data))

    def loss_fn(self, predict, target):
        return IS_divergence(predict, target)


class NMF_Beta(NMF_L2):
    def __init__(self, *args, beta=2):
        self.b = beta
        super().__init__(*args)

    def get_W_positive(self):
        return (self.W.data @ self.H.data) ** (self.b - 1) @ self.H.data.t()

    def get_H_positive(self):
        return self.W.data.t() @ (self.W.data @ self.H.data) ** (self.b - 1)

    def loss_fn(self, predict, target):
        return Beta_divergence(predict, target, beta=self.b)


class _NMFD(_NMF):
    """
    NMF deconvolution model.
    """

    def __init__(self, Vshape, R, T=1, W=None, H=None, fix_W=False, fix_H=False):
        """

        :param Vshape: Shape of the target matrix.
        :type Vshape: tuple.

        :param T: Size of the templates.
        :param R: Number of components.
        :param W: A torch.Tensor or np.ndarray as initial value or None.
        :param H: A torch.Tensor or np.ndarray as initial value or None.
        :param fix_W: Whether to fix W or not.
        :param fix_H: Whether to fix H or not.
        """
        self.T = T
        K, M = Vshape
        if W is None:
            W = (K, R, T)
        if H is None:
            H = (R, M)
        super().__init__(K, M, R, W, H, fix_W, fix_H)

    def conv(self, w, h):
        return F.conv1d(h[None, :], w.flip(2), padding=self.T - 1)[0, :, :self.M]

    def reconstruct(self):
        return self.conv(self.W, self.H)


class NMFD_KL(_NMFD):
    """
    NMF deconvolution model.
    """

    def get_W_positive(self):
        return self.H.data.sum(1, keepdim=True).t()[..., None]

    def get_H_positive(self):
        return self.W.data.sum(0).sum(1, keepdim=True)

    def loss_fn(self, predict, target):
        return KL_divergence(predict, target)


class NMFD_L2(_NMFD):
    """
    NMF deconvolution model.
    """

    def get_W_positive(self):
        expand_H = torch.stack([F.pad(self.H.data[:, :self.M - j], (j, 0)) for j in range(self.T)], dim=2)
        tmp = self.conv(self.W.data, self.H.data) @ expand_H
        return tmp.transpose(0, 1)

    def get_H_positive(self):
        V = self.conv(self.W.data, self.H.data)
        expand_V = torch.stack([F.pad(V[:, j:], (0, j)) for j in range(self.T)], dim=0)
        return (self.W.data.transpose(0, 2) @ expand_V).sum(0)

    def loss_fn(self, predict, target):
        return Euclidean(predict, target)
