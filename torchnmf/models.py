from torch import nn
from torch.nn import functional as F
from .utils import *

print("brachn test")


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
        self.fix_W = False
        self.fix_H = False
        if type(W) == tuple:
            init_W = torch.Tensor(*W)
        else:
            init_W = return_torch(W)
            self.fix_W = fix_W

        if type(H) == tuple:
            init_H = torch.Tensor(*H)
        else:
            init_H = return_torch(H)
            self.fix_H = fix_H

        self.W = nn.Parameter(init_W, requires_grad=False)
        self.H = nn.Parameter(init_H, requires_grad=False)
        self.reset_weight()

    def reset_weight(self):
        """
        Initialize the value of weight equally sample from 0~1 when it's not fixed.

        :return: None.
        """
        if not self.fix_W:
            self.W.data.uniform_()
        if not self.fix_H:
            self.H.data.uniform_()

    def forward(self, V, n_iter=1):
        """
        Fit the model based on target matrix V.

        :param V: Target matrix.
        :param n_iter: Number of iterations.
        :return: W, H.
        """
        for i in range(n_iter):
            VonV = V / self.reconstruct()
            modified = False
            if not self.fix_W:
                self._update_W(VonV)
                modified = True
            if not self.fix_H:
                if modified:
                    VonV = V / self.reconstruct()
                self._update_H(VonV)
        return self.W, self.H

    def _update_W(self, VV):
        """
        Update mechanism for Basis Matrix W.

        :param VV: Element-wise ratio of the target matrix and the reconstructed matrix.
        :return: None
        """

    def _update_H(self, VV):
        """
        Update mechanism for Mixture Matrix H.

        :param VV: Element-wise ratio of the target matrix and the reconstructed matrix.
        :return: None:
        """

    def reconstruct(self, W=None, H=None):
        """
        Reconstruct mechanism for target matrix.

        :param W: Basis matrix. If not provided will use internal matrix.
        :param H: Mixture matrix. If not provided will use internal matrix.
        :return: Reconstructed matrix.
        """
        if W is None:
            W = self.W
        if H is None:
            H = self.H
        return self._reconstruct(W, H)

    def _reconstruct(self, W, H):
        return W @ H


class NMF(_NMF):
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

    def _update_W(self, VV):
        Ht = self.H.t()
        self.W *= VV @ Ht / Ht.sum(0)

    def _update_H(self, VV):
        Wt = self.W.t()
        self.H *= Wt @ VV / Wt.sum(1, keepdim=True)


class NMFD(_NMF):
    """
    NMF deconvolution model.
    """

    def __init__(self, Vshape, R, T=5, W=None, H=None, fix_W=False, fix_H=False):
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

    def _update_W(self, VV):
        expand_H = torch.stack([F.pad(self.H[:, :self.M - j], (j, 0)) for j in range(self.T)], dim=2)
        upper = VV @ expand_H
        lower = expand_H.sum(1, keepdim=True)
        self.W *= (upper / lower).transpose(0, 1)

    def _update_H(self, VV):
        expand_SonV = torch.stack([F.pad(VV[:, j:], (0, j)) for j in range(self.T)], dim=0)
        Wt = self.W.transpose(0, 2)
        upper = Wt @ expand_SonV  # (T, R, M)
        lower = Wt.sum(2, keepdim=True)
        self.H *= torch.mean(upper / lower, 0)

    def _reconstruct(self, W, H):
        return F.conv1d(H[None, :], W.flip(2), padding=self.T - 1)[0, :, :self.M]
