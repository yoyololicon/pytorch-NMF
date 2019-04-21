import torch
from .base import Base
from .utils import normalize
from tqdm import tqdm


def _log_probability(V, WZH, W, Z, H, W_alpha, Z_alpha, H_alpha):
    return V.view(-1) @ WZH.log().view(-1) + W.mul(W_alpha - 1).log().sum() + H.mul(H_alpha - 1).log().sum() + \
           Z.mul(Z_alpha - 1).log().sum()


class PLCA(Base):

    def __init__(self, Xshape: tuple, rank: int = None, uniform=False):
        """

        :param Xshape: Target matrix size.
        :param n_components:
        """
        super().__init__()
        self.K, self.M = Xshape
        if not rank:
            self.rank = self.K
        else:
            self.rank = rank

        self.W = torch.nn.Parameter(normalize(torch.rand(self.K, self.rank), 0), requires_grad=False)
        self.H = torch.nn.Parameter(normalize(torch.rand(self.rank, self.M), 1), requires_grad=False)
        self.Z = torch.nn.Parameter(normalize(torch.rand(self.rank)), requires_grad=False)

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
            self.Z[:] = normalize(self.fix_neg(Z), 0)

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

                self.update_params(X / V, update_W, update_H, update_Z, W_alpha, Z_alpha, H_alpha)
                if verbose:
                    pbar.set_postfix(Log_likelihood=log_prob)
                    #pbar.set_description('Log likelihood=%.4f' % log_prob)
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

    def __init__(self, Xshape, T=1, n_components=None):
        """

        :param Xshape: Target matrix size.
        :param n_components:
        """
        super().__init__(Xshape, n_components)
        self.T = T
        self.W = torch.nn.Parameter(torch.Tensor(self.K, self.rank, self.T), requires_grad=False)
        self.H = torch.nn.Parameter(torch.Tensor(self.rank, self.M - self.T + 1), requires_grad=False)

    def forward(self, H=None, W=None, Z=None):
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        if Z is None:
            Z = self.Z
        return F.conv1d(H[None, :], W.flip(2) * Z[:, None], padding=self.T - 1)[0]

    def _initialize(self):
        super()._initialize()
        self.W[:] = self._normalize(self.W, [0, 2])

    def _m_step(self, PonWZH, update_W, update_H, update_Z, W_alpha, H_alpha, Z_alpha):
        # type: (Tensor, bool, bool, bool, float, float, float) -> None
        new_W = F.conv1d(self.H[:, None], PonWZH[:, None], padding=self.T - 1).transpose(0, 1).flip(2) * self.W
        Z = new_W.sum((0, 2))
        if update_W:
            new_W /= Z[:, None]
            new_W = self.fix_neg(new_W + W_alpha - 1)
            self.W[:] = self._normalize(new_W, [0, 2])
        if update_H:
            PonWZH = F.pad(PonWZH, [0, self.T - 1])
            new_H = F.conv1d(PonWZH[None, ...], torch.transpose(self.W * self.Z[:, None], 0, 1))[0] * self.H
            new_H = self.fix_neg(self._normalize(new_H, [1]) + H_alpha - 1)
            self.H[:] = self._normalize(new_H, [1])
        if update_Z:
            Z += Z_alpha - 1
            self.Z[:] = self._normalize(self.fix_neg(Z), [0])

    def sort(self):
        _, maxidx = self.W.data.sum(2).max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        self.H.data = self.H.data[idx]
        self.Z.data = self.Z.data[idx]


class SIPLCA2(SIPLCA):

    def __init__(self, Xshape, win=1, n_components=None):
        """

        :param Xshape: Target matrix size.
        :param n_components:
        """
        try:
            F, T = win
        except:
            F = T = win
        super().__init__(Xshape, T, n_components)
        self.F = F
        self.pad_size = (self.F - 1, self.T - 1)
        self.W = nn.Parameter(torch.Tensor(self.F, self.rank, self.T), requires_grad=False)
        self.H = nn.Parameter(torch.Tensor(self.rank, Xshape[0] - F + 1, Xshape[1] - T + 1),
                              requires_grad=False)

    def forward(self, H=None, W=None, Z=None):
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        if Z is None:
            Z = self.Z
        return F.conv2d(H[None, ...], W.mul(Z[:, None]).transpose(0, 1).flip((1, 2))[None, ...], padding=self.pad_size)[
            0, 0]

    def _initialize(self):
        super()._initialize()
        self.H.data.fill_(1 / self.K / self.M)

    def _m_step(self, PonWZH, update_W, update_H, update_Z, W_alpha, H_alpha, Z_alpha):
        # type: (Tensor, bool, bool, bool, float, float, float) -> None
        PonWZH = PonWZH[None, None, ...]
        new_W = F.conv2d(
            self.H.mul(self.Z[:, None, None])[:, None], PonWZH, padding=self.pad_size).squeeze(1).flip(
            (1, 2)).transpose(0, 1) * self.W
        Z = new_W.sum((0, 2))
        if update_W:
            new_W /= Z[:, None]
            new_W = self.fix_neg(new_W + W_alpha - 1)
            self.W[:] = self._normalize(new_W, [0, 2])
        if update_H:
            new_H = F.conv2d(PonWZH, torch.transpose(self.W * self.Z[:, None], 0, 1)[:, None])[0] * self.H
            new_H = self.fix_neg(self._normalize(new_H, [1]) + H_alpha - 1)
            self.H[:] = self._normalize(new_H, [1, 2])
        if update_Z:
            Z += Z_alpha - 1
            self.Z[:] = self._normalize(self.fix_neg(Z), [0])

    def sort(self):
        _, maxidx = self.W.data.sum(2).max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        self.H.data = self.H.data[idx]
        self.Z.data = self.Z.data[idx]
