from torch import nn
from torch.nn import functional as F
from .utils import *


class _NMF(nn.Module):
    def __init__(self, K, M, R, W, H):
        super().__init__()
        self.K = K
        self.M = M
        self.R = R
        self.fix_W = True
        self.fix_H = True
        if type(W) == tuple:
            init_W = torch.Tensor(*W)
            self.fix_W = False
        else:
            init_W = return_torch(W)

        if type(H) == tuple:
            init_H = torch.Tensor(*H)
            self.fix_H = False
        else:
            init_H = return_torch(H)

        self.W = nn.Parameter(init_W, requires_grad=False)
        self.H = nn.Parameter(init_H, requires_grad=False)
        self.reset_weight()

    def reset_weight(self):
        if not self.fix_W:
            self.W.data.uniform_()
        if not self.fix_H:
            self.H.data.uniform_()

    def forward(self, V, n_iter=1):
        for i in range(n_iter):
            V_tile = self.reconstruct(self.W, self.H)
            VonV = V / V_tile
            if not self.fix_W:
                self.update_W(VonV)
            if not self.fix_H:
                self.update_H(VonV)
        return self.W, self.H

    def update_W(self, VV):
        """

        :return:
        """

    def update_H(self, VV):
        """

        :param VV:
        :return:
        """

    def reconstruct(self, W, H):
        """

        :return:
        """
        return W @ H


class NMF(_NMF):
    """
    doc.
    """

    def __init__(self, Vshape, R, W=None, H=None):
        K, M = Vshape
        if W is None:
            W = (K, R)
        if H is None:
            H = (R, M)
        super().__init__(K, M, R, W, H)

    def update_W(self, VV):
        Ht = self.H.t()
        self.W *= VV @ Ht / Ht.sum(0)

    def update_H(self, VV):
        Wt = self.W.t()
        self.H *= Wt @ VV / Wt.sum(1, keepdim=True)


class NMFD(_NMF):
    """
    doc.
    """

    def __init__(self, Vshape, R, T=5, W=None, H=None):
        self.T = T
        K, M = Vshape
        if W is None:
            W = (K, R, T)
        if H is None:
            H = (R, M)
        super().__init__(K, M, R, W, H)

    def update_W(self, VV):
        expand_H = torch.stack([F.pad(self.H[:, :self.M - j], (j, 0)) for j in range(self.T)], dim=2)
        upper = (VV @ expand_H)
        lower = expand_H.sum(1, keepdim=True)
        self.W *= (upper / lower).transpose(0, 1)

    def update_H(self, VV):
        expand_SonV = torch.stack([F.pad(VV[:, j:], (0, j)) for j in range(self.T)], dim=0)
        Wt = self.W.transpose(0, 2)
        upper = Wt @ expand_SonV  # (T, R, M)
        lower = Wt.sum(2, keepdim=True)
        self.H *= torch.mean(upper / lower, 0)

    def reconstruct(self, W, H):
        return F.conv1d(H[None, :], W.flip(2), padding=self.T - 1)[0, :, :self.M]
