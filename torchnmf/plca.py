import torch
import torch.nn.functional as F
from torch.nn import Parameter
from .base import Base
from .utils import normalize
from tqdm import tqdm
from .metrics import KL_divergence


def _log_probability(V, WZH, W, Z, H, W_alpha, Z_alpha, H_alpha):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float) -> Tensor
    return V.view(-1) @ WZH.log().view(-1) + W.log().sum().mul(W_alpha - 1) + H.log().sum().mul(
        H_alpha - 1) + Z.log().sum().mul(Z_alpha - 1)


class PLCA_(Base):
    def __init__(self, W_size, Z_size, H_size, W_norm_dim, H_norm_dim, uniform):
        super().__init__()
        self.rank = Z_size
        self.W = Parameter(normalize(torch.rand(*W_size), W_norm_dim), requires_grad=False)
        self.H = Parameter(normalize(torch.rand(*H_size), H_norm_dim), requires_grad=False)
        self.Z = Parameter(normalize(torch.rand(self.rank)), requires_grad=False)

        if uniform:
            self.H.data.copy_(normalize(torch.ones(*H_size), H_norm_dim))
            self.Z.data.fill_(1 / self.rank)

    def forward(self, H=None, W=None, Z=None):
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        if Z is None:
            Z = self.Z
        return self.reconstruct(W, Z, H)

    def reconstruct(self, W, Z, H):
        raise NotImplementedError

    def fit(self,
            V,
            W=None,
            Z=None,
            H=None,
            update_W=True,
            update_Z=True,
            update_H=True,
            tol=1e-5,
            max_iter=40,
            verbose=1,
            W_alpha=1,
            Z_alpha=1,
            H_alpha=1
            ):

        norm = V.sum()
        V = V / norm

        if W is not None:
            self.W.data.copy_(W)

        if H is not None:
            self.H.data.copy_(H)

        if Z is not None:
            self.Z.data.copy_(Z)

        with tqdm(total=max_iter, disable=not verbose) as pbar:
            for n_iter in range(1, max_iter + 1):
                WZH = self.reconstruct(self.W, self.Z, self.H)

                log_prob = _log_probability(V, WZH, self.W, self.Z, self.H, W_alpha, Z_alpha, H_alpha).item()
                kl_div = KL_divergence(WZH, V).item()

                self.update_params(V / WZH, update_W, update_H, update_Z, W_alpha, Z_alpha, H_alpha)

                pbar.set_postfix(Log_likelihood=log_prob, KL_divergence=kl_div)
                # pbar.set_description('Log likelihood=%.4f' % log_prob)
                pbar.update()

        return n_iter, norm

    def fit_transform(self, *args, **kwargs):
        n_iter, norm = self.fit(*args, **kwargs)
        return n_iter, self.forward() * norm

    def update_params(self, VdivWZH, update_W, update_H, update_Z, W_alpha, H_alpha, Z_alpha):
        # type: (Tensor, bool, bool, bool, float, float, float) -> None
        raise NotImplementedError


class PLCA(PLCA_):

    def __init__(self, Vshape: tuple, rank: int = None, uniform=False):
        self.K, self.M = Vshape
        if not rank:
            rank = self.K
        super().__init__((self.K, rank), rank, (rank, self.M), (0,), (1,), uniform)

    def reconstruct(self, W, Z, H):
        return (W * Z) @ H

    def update_params(self, VdivWZH, update_W, update_H, update_Z, W_alpha, H_alpha, Z_alpha):
        # type: (Tensor, bool, bool, bool, float, float, float) -> None
        tmp = torch.unsqueeze(self.W * self.Z, 2) * self.H * VdivWZH[:, None]
        if update_W:
            W = self.fix_neg(tmp.sum(2) + W_alpha - 1)
            self.W[:] = normalize(W, 0)
        if update_H:
            H = self.fix_neg(tmp.sum(0) + H_alpha - 1)
            self.H[:] = normalize(H, 1)
        if update_Z:
            self.Z[:] = normalize(self.fix_neg(torch.sum(tmp, (0, 2)) + Z_alpha - 1))

    def sort(self):
        _, maxidx = self.W.data.max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        self.H.data = self.H.data[idx]
        self.Z.data = self.Z.data[idx]


class SIPLCA(PLCA_):

    def __init__(self, Vshape: tuple, rank: int = None, T: int = 1, uniform=False):
        self.K, self.M = Vshape
        self.pad_size = T - 1
        if not rank:
            rank = self.K

        W_size = (self.K, rank, T)
        H_size = (rank, self.M - T + 1)
        H_norm_dim = (1,)
        super().__init__(W_size, rank, H_size, (0, 2), H_norm_dim, uniform)

    def reconstruct(self, W, Z, H):
        return F.conv1d(H[None, ...], W.flip(2) * Z[:, None], padding=self.pad_size)[0]

    def update_params(self, VdivWZH, update_W, update_H, update_Z, W_alpha, H_alpha, Z_alpha):
        # type: (Tensor, bool, bool, bool, float, float, float) -> None

        if update_W or update_Z:
            new_W = F.conv1d(VdivWZH[:, None], self.H[:, None] * self.Z[:, None, None]) * self.W

        if update_H:
            new_H = F.conv1d(VdivWZH[None, ...], torch.transpose(self.W * self.Z[:, None], 0, 1))[0] * self.H
            new_H = normalize(self.fix_neg(new_H + H_alpha - 1), 1)
            self.H[:] = new_H

        if update_W:
            new_W = normalize(self.fix_neg(new_W + W_alpha - 1), (0, 2))
            self.W[:] = new_W

        if update_Z:
            Z = normalize(self.fix_neg(new_W.sum((0, 2)) + Z_alpha - 1))
            self.Z[:] = Z

    def sort(self):
        _, maxidx = self.W.data.sum(2).max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        self.H.data = self.H.data[idx]
        self.Z.data = self.Z.data[idx]


class SIPLCA2(PLCA_):

    def __init__(self, Vshape: tuple, rank: int = None, win=1, uniform=False):
        try:
            F, T = win
        except:
            F = T = win
        if len(Vshape) == 3:
            self.channel, self.K, self.M = Vshape
        else:
            self.K, self.M = Vshape
            self.channel = 1

        self.pad_size = (F - 1, T - 1)
        if not rank:
            rank = self.K

        W_size = (self.channel, rank, F, T)
        H_size = (rank, self.K - F + 1, self.M - T + 1)
        W_norm_dim = (0, 2, 3)
        H_norm_dim = (1, 2)
        super().__init__(W_size, rank, H_size, W_norm_dim, H_norm_dim, uniform)

    def reconstruct(self, W, Z, H):
        out = F.conv2d(H[None, ...], W.mul(Z[:, None, None]).flip((2, 3)), padding=self.pad_size)[0]
        if self.channel == 1:
            return out[0]
        return out

    def update_params(self, VdivWZH, update_W, update_H, update_Z, W_alpha, H_alpha, Z_alpha):
        # type: (Tensor, bool, bool, bool, float, float, float) -> None
        VdivWZH = VdivWZH.view(self.channel, 1, self.K, self.M)
        if update_W or update_Z:
            new_W = F.conv2d(VdivWZH, self.H.mul(self.Z[:, None, None])[:, None]) * self.W

        if update_H:
            new_H = F.conv2d(VdivWZH.transpose(0, 1), torch.transpose(self.W * self.Z[:, None, None], 0, 1))[0] * self.H
            new_H = normalize(self.fix_neg(new_H + H_alpha - 1), (1, 2))
            self.H[:] = new_H

        if update_W:
            new_W = normalize(self.fix_neg(new_W + W_alpha - 1), (0, 2, 3))
            self.W[:] = new_W

        if update_Z:
            Z = normalize(self.fix_neg(new_W.sum((0, 2, 3)) + Z_alpha - 1))
            self.Z[:] = Z

    def sort(self):
        raise NotImplementedError


class SIPLCA3(PLCA_):
    def __init__(self, Vshape: tuple, rank: int = None, win=1, uniform=False):
        try:
            T, H, W = win
        except:
            T = H = W = win
        if len(Vshape) == 4:
            self.channel, self.N, self.K, self.M = Vshape
        else:
            self.N, self.K, self.M = Vshape
            self.channel = 1

        self.pad_size = (T - 1, H - 1, W - 1)
        if not rank:
            rank = self.K

        W_size = (self.channel, rank, T, H, W)
        H_size = (rank, self.N - T + 1, self.K - H + 1, self.M - W + 1)
        W_norm_dim = (0, 2, 3, 4)
        H_norm_dim = (1, 2, 3)
        super().__init__(W_size, rank, H_size, W_norm_dim, H_norm_dim, uniform)

    def reconstruct(self, W, Z, H):
        out = F.conv3d(H[None, ...], W.mul(Z[:, None, None, None]).flip((2, 3, 4)), padding=self.pad_size)[0]
        if self.channel == 1:
            return out[0]
        return out

    def update_params(self, VdivWZH, update_W, update_H, update_Z, W_alpha, H_alpha, Z_alpha):
        # type: (Tensor, bool, bool, bool, float, float, float) -> None
        VdivWZH = VdivWZH.view(self.channel, 1, self.N, self.K, self.M)
        if update_W or update_Z:
            new_W = F.conv3d(VdivWZH, self.H.mul(self.Z[:, None, None, None])[:, None], padding=self.pad_size) * self.W

        if update_H:
            new_H = F.conv3d(VdivWZH.transpose(0, 1), torch.transpose(self.W * self.Z[:, None, None, None], 0, 1))[
                        0] * self.H
            new_H = normalize(self.fix_neg(new_H + H_alpha - 1), (1, 2, 3))
            self.H[:] = new_H

        if update_W:
            new_W = normalize(self.fix_neg(new_W + W_alpha - 1), (0, 2, 3, 4))
            self.W[:] = new_W

        if update_Z:
            Z = normalize(self.fix_neg(new_W.sum((0, 2, 3, 4)) + Z_alpha - 1))
            self.Z[:] = Z

    def sort(self):
        raise NotImplementedError
