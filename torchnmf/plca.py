import torch
import torch.nn.functional as F
from .base import Base
from .utils import normalize
from tqdm import tqdm
from .metrics import KL_divergence


def _log_probability(V, WZH, W, Z, H, W_alpha, Z_alpha, H_alpha):
    return V.view(-1) @ WZH.log().view(-1) + W.mul(W_alpha - 1).log().sum() + H.mul(H_alpha - 1).log().sum() + \
           Z.mul(Z_alpha - 1).log().sum()


class PLCA(Base):

    def __init__(self, Xshape: tuple, rank: int = None, uniform=False):
        super().__init__()
        self.K, self.M = Xshape
        if not rank:
            self.rank = self.K
        else:
            self.rank = rank

        self.W = torch.nn.Parameter(normalize(torch.rand(self.K, self.rank), 0), requires_grad=False)
        self.H = torch.nn.Parameter(normalize(torch.rand(self.rank, self.M), 1), requires_grad=False)
        self.Z = torch.nn.Parameter(normalize(torch.rand(self.rank)), requires_grad=False)
        self.kl_scale = self.K * self.M

        if uniform:
            self.H.data.fill_(1 / self.M)
            self.Z.data.fill_(1 / self.rank)

    def forward(self, H=None, W=None, Z=None):
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        if Z is None:
            Z = self.Z
        return (W * Z) @ H

    def update_params(self, VdivWZH, update_W, update_H, update_Z, W_alpha, H_alpha, Z_alpha):
        # type: (Tensor, bool, bool, bool, float, float, float) -> None
        tmp = torch.unsqueeze(self.W * self.Z, 2) * self.H * VdivWZH[:, None]
        Z = torch.sum(tmp, (0, 2))
        if update_W:
            W = self.fix_neg(tmp.sum(2) / Z + W_alpha - 1)
            self.W[:] = normalize(W, 0)
        if update_H:
            H = self.fix_neg(tmp.sum(0) / Z[:, None] + H_alpha - 1)
            self.H[:] = normalize(H, 1)
        if update_Z:
            Z += Z_alpha - 1
            self.Z[:] = normalize(self.fix_neg(Z))

    def fit(self,
            X,
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
        norm = X.sum()
        X = X / norm

        if W is not None:
            self.W.data.copy_(W)

        if H is not None:
            self.H.data.copy_(H)

        if Z is not None:
            self.Z.data.copy_(Z)

        with tqdm(total=max_iter) as pbar:
            for n_iter in range(1, max_iter + 1):
                V = self.forward()

                log_prob = _log_probability(X, V, self.W, self.Z, self.H, W_alpha, Z_alpha, H_alpha)
                kl_div = KL_divergence(V, X) / self.kl_scale

                self.update_params(X / V, update_W, update_H, update_Z, W_alpha, Z_alpha, H_alpha)
                if verbose:
                    pbar.set_postfix(Log_likelihood=log_prob, KL_divergence=kl_div)
                    # pbar.set_description('Log likelihood=%.4f' % log_prob)
                    pbar.update()

        return n_iter, norm

    def fit_transform(self, *args, **kwargs):
        n_iter, norm = self.fit(*args, **kwargs)
        return n_iter, self.forward() * norm

    def sort(self):
        _, maxidx = self.W.data.max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        self.H.data = self.H.data[idx]
        self.Z.data = self.Z.data[idx]


class SIPLCA(PLCA):

    def __init__(self, Xshape: tuple, rank: int = None, T: int = 1, uniform=False):
        Base.__init__(self)
        self.K, self.M = Xshape
        if not rank:
            self.rank = self.K
        else:
            self.rank = rank
        self.T = T

        self.W = torch.nn.Parameter(normalize(torch.rand(self.K, self.rank, self.T), (0, 2)), requires_grad=False)
        self.H = torch.nn.Parameter(normalize(torch.rand(self.rank, self.M - self.T + 1), 1), requires_grad=False)
        self.Z = torch.nn.Parameter(normalize(torch.rand(self.rank)), requires_grad=False)

        self.kl_scale = self.K * self.M

        if uniform:
            self.H.data.fill_(1 / self.M)
            self.Z.data.fill_(1 / self.rank)

    def forward(self, H=None, W=None, Z=None):
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        if Z is None:
            Z = self.Z
        if len(H.shape) == 2:
            H = H[None, ...]
        return F.conv1d(H[None, ...], W.flip(2) * Z[:, None], padding=self.T - 1)[0]

    def update_params(self, VdivWZH, update_W, update_H, update_Z, W_alpha, H_alpha, Z_alpha):
        # type: (Tensor, bool, bool, bool, float, float, float) -> None
        if update_W or update_Z:
            new_W = F.conv1d(self.H[:, None], VdivWZH[:, None], padding=self.T - 1).transpose(0, 1).flip(2) * self.W
            Z = new_W.sum((0, 2))

        if update_H:
            new_H = F.conv1d(VdivWZH[None, ...], torch.transpose(self.W * self.Z[:, None], 0, 1))[0] * self.H
            new_H = normalize(new_H, 1)
            if H_alpha != 1:
                new_H = normalize(self.fix_neg(new_H + H_alpha - 1), 1)
            self.H[:] = new_H

        if update_W:
            new_W /= Z[:, None]
            if W_alpha != 1:
                new_W = normalize(self.fix_neg(new_W + W_alpha - 1), (0, 2))
            self.W[:] = new_W

        if update_Z:
            if Z_alpha != 1:
                Z = normalize(self.fix_neg(Z + Z_alpha - 1))
            self.Z[:] = Z

    def sort(self):
        _, maxidx = self.W.data.sum(2).max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        self.H.data = self.H.data[idx]
        self.Z.data = self.Z.data[idx]


class SIPLCA2(SIPLCA):

    def __init__(self, Xshape: tuple, rank: int = None, win=1, uniform=False):
        Base.__init__(self)
        try:
            self.F, self.T = win
        except:
            self.F = self.T = win
        if len(Xshape) == 3:
            self.channel, self.K, self.M = Xshape
        else:
            self.K, self.M = Xshape
            self.channel = 1

        if not rank:
            self.rank = self.K
        else:
            self.rank = rank

        self.W = torch.nn.Parameter(normalize(torch.rand(self.channel, self.rank, self.F, self.T), (0, 2, 3)),
                                    requires_grad=False)
        self.H = torch.nn.Parameter(normalize(torch.rand(self.rank, self.K - self.F + 1, self.M - self.T + 1), (1, 2)),
                                    requires_grad=False)
        self.Z = torch.nn.Parameter(normalize(torch.rand(self.rank)), requires_grad=False)
        self.kl_scale = self.K * self.M * self.channel
        self.pad_size = (self.F - 1, self.T - 1)

        if uniform:
            self.H.data.fill_(1 / (self.M - self.T + 1) / (self.K - self.F + 1))
            self.Z.data.fill_(1 / self.rank)

    def forward(self, H=None, W=None, Z=None):
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        if Z is None:
            Z = self.Z
        out = F.conv2d(H[None, ...], W.mul(Z[:, None, None]).flip((2, 3)), padding=self.pad_size)[0]
        if self.channel == 1:
            return out[0]
        return out

    def update_params(self, VdivWZH, update_W, update_H, update_Z, W_alpha, H_alpha, Z_alpha):
        # type: (Tensor, bool, bool, bool, float, float, float) -> None
        if update_W or update_Z:
            VdivWZH = VdivWZH.view(self.channel, 1, self.K, self.M)
            new_W = F.conv2d(self.H.mul(self.Z[:, None, None])[:, None], VdivWZH, padding=self.pad_size).flip(
                (2, 3)).transpose(0, 1) * self.W
            Z = new_W.sum((0, 2, 3))

        if update_H:
            new_H = F.conv2d(VdivWZH.transpose(0, 1), torch.transpose(self.W * self.Z[:, None, None], 0, 1))[0] * self.H
            new_H = normalize(new_H, (1, 2))
            if H_alpha != 1:
                new_H = normalize(self.fix_neg(new_H + H_alpha - 1), (1, 2))
            self.H[:] = new_H

        if update_W:
            new_W /= Z[:, None, None]
            if W_alpha != 1:
                new_W = normalize(self.fix_neg(new_W + W_alpha - 1), (0, 2, 3))
            self.W[:] = new_W

        if update_Z:
            if Z_alpha != 1:
                Z = normalize(self.fix_neg(Z + Z_alpha - 1))
            self.Z[:] = Z


class SIPLCA3(PLCA):

    def __init__(self, Xshape: tuple, rank: int = None, win=1, uniform=False):
        Base.__init__(self)
        try:
            self.D, self.F, self.T = win
        except:
            self.D = self.F = self.T = win
        if len(Xshape) == 4:
            self.channel, self.N, self.K, self.M = Xshape
        else:
            self.N, self.K, self.M = Xshape
            self.channel = 1

        if not rank:
            self.rank = self.K
        else:
            self.rank = rank

        self.W = torch.nn.Parameter(
            normalize(torch.rand(self.channel, self.rank, self.D, self.F, self.T), (0, 2, 3, 4)), requires_grad=False)
        self.H = torch.nn.Parameter(
            normalize(torch.rand(self.rank, self.N - self.D + 1, self.K - self.F + 1, self.M - self.T + 1), (1, 2, 3)),
            requires_grad=False)
        self.Z = torch.nn.Parameter(normalize(torch.rand(self.rank)), requires_grad=False)
        self.kl_scale = self.K * self.M * self.N * self.channel
        self.pad_size = (self.D - 1, self.F - 1, self.T - 1)

        if uniform:
            self.H.data.copy_(
                normalize(torch.ones(self.rank, self.N - self.D + 1, self.K - self.F + 1, self.M - self.T + 1),
                          (1, 2, 3)))
            self.Z.data.fill_(1 / self.rank)

    def forward(self, H=None, W=None, Z=None):
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        if Z is None:
            Z = self.Z
        out = F.conv3d(H[None, ...], W.mul(Z[:, None, None, None]).flip((2, 3, 4)), padding=self.pad_size)[0]
        if self.channel == 1:
            return out[0]
        return out

    def update_params(self, VdivWZH, update_W, update_H, update_Z, W_alpha, H_alpha, Z_alpha):
        # type: (Tensor, bool, bool, bool, float, float, float) -> None
        if update_W or update_Z:
            VdivWZH = VdivWZH.view(self.channel, 1, self.N, self.K, self.M)
            new_W = F.conv3d(self.H.mul(self.Z[:, None, None, None])[:, None], VdivWZH, padding=self.pad_size).flip(
                (2, 3, 4)).transpose(0, 1) * self.W
            Z = new_W.sum((0, 2, 3, 4))

        if update_H:
            new_H = F.conv3d(VdivWZH.transpose(0, 1), torch.transpose(self.W * self.Z[:, None, None, None], 0, 1))[
                        0] * self.H
            new_H = normalize(new_H, (1, 2, 3))
            if H_alpha != 1:
                new_H = normalize(self.fix_neg(new_H + H_alpha - 1), (1, 2, 3))
            self.H[:] = new_H

        if update_W:
            new_W /= Z[:, None, None, None]
            if W_alpha != 1:
                new_W = normalize(self.fix_neg(new_W + W_alpha - 1), (0, 2, 3, 4))
            self.W[:] = new_W

        if update_Z:
            if Z_alpha != 1:
                Z = normalize(self.fix_neg(Z + Z_alpha - 1))
            self.Z[:] = Z

    def sort(self):
        raise NotImplementedError
