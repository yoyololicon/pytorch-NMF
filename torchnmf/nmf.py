import torch
import torch.nn.functional as F
from torch.nn import Parameter
from .metrics import Beta_divergence
from .base import Base
from .utils import renorm_
from tqdm import tqdm
from collections.abc import Iterable


def _mu_update(param: torch.Tensor, pos: torch.Tensor, gamma: float,
               l1_reg: torch.Tensor, l2_reg: torch.Tensor,
               reg_index: torch.LongTensor, reg_dim: int):
    if param.grad is None:
        return
    # prevent negative term, very likely to happen with kl divergence
    multiplier = F.relu(pos - param.grad, inplace=True)

    if reg_index is not None:
        if l1_reg > 0:
            # pos.add_(l1_reg)
            pos.index_add_(reg_dim, reg_index, torch.index_select(l1_reg.expand_as(pos), reg_dim, reg_index))
        if l2_reg > 0:
            if pos.shape != param.shape:
                pos = pos.expand_as(param).index_add(reg_dim, reg_index,
                                                     l2_reg * param.index_select(reg_dim, reg_index))
                # pos = pos + l2_reg * param
            else:
                pos.index_add_(reg_dim, reg_index, l2_reg * param.index_select(reg_dim, reg_index))
    multiplier.div_(pos)
    if gamma != 1:
        multiplier.pow_(gamma)
    param.mul_(multiplier)


class _NMF(Base):
    def __init__(self, W_size, H_size, rank):
        super().__init__()
        self.rank = rank
        self.W = Parameter(torch.rand(*W_size))
        self.H = Parameter(torch.rand(*H_size))

    def forward(self, H=None, W=None):
        if H is None:
            H = self.H
        if W is None:
            W = self.W
        return self.reconstruct(H, W)

    def reconstruct(self, H, W):
        raise NotImplementedError

    def get_W_positive(self, WH, beta, H_sum) -> (torch.Tensor, None or torch.Tensor):
        raise NotImplementedError

    def get_H_positive(self, WH, beta, W_sum) -> (torch.Tensor, None or torch.Tensor):
        raise NotImplementedError

    def fit(self,
            V,
            W=None,
            H=None,
            update_W=True,
            update_H=True,
            beta=1,
            tol=1e-5,
            max_iter=200,
            verbose=0,
            initial='random',
            alpha=0,
            l1_ratio=0,
            W_reg_control=None,
            H_reg_control=None,
            ):

        V = self.fix_neg(V)

        if W is None:
            pass  # will do special initialization in thre future
        else:
            self.W.data.copy_(W)
            self.W.requires_grad = update_W

        if H is None:
            pass
        else:
            self.H.data.copy_(H)
            self.H.requires_grad = update_H

        if beta < 1:
            gamma = 1 / (2 - beta)
        elif beta > 2:
            gamma = 1 / (beta - 1)
        else:
            gamma = 1

        l1_reg = torch.Tensor([alpha * l1_ratio]).to(V.device)
        l2_reg = alpha * (1 - l1_ratio)

        # construct sparsity control mask
        W_reg_index = H_reg_index = torch.arange(self.rank, device=self.W.device, dtype=torch.long)
        if type(W_reg_control) is float or type(W_reg_control) is int:
            assert 0 <= W_reg_control <= 1, "regularization ratio should be between zero and one"
            W_reg_index = torch.arange(round(self.rank * W_reg_control), device=self.W.device, dtype=torch.long)
        elif isinstance(W_reg_control, Iterable):
            W_reg_index = torch.LongTensor(W_reg_control).to(self.W.device)
        elif W_reg_control is not None:
            raise ValueError("W_reg_index should be an index array or ratio between zero and one!")

        if type(H_reg_control) is float or type(H_reg_control) is int:
            assert 0 <= H_reg_control <= 1, "regularization ratio should be between zero and one"
            H_reg_index = torch.arange(round(self.rank * H_reg_control), device=self.H.device)
        elif isinstance(H_reg_control, Iterable):
            H_reg_index = torch.LongTensor(H_reg_control).to(self.H.device)
        elif H_reg_control is not None:
            raise ValueError("H_reg_index should be an index array or ratio between zero and one!")

        loss_scale = torch.prod(torch.tensor(V.shape)).float()

        H_sum, W_sum = None, None
        with tqdm(total=max_iter, disable=not verbose) as pbar:
            for n_iter in range(max_iter):
                if self.W.requires_grad:
                    self.zero_grad()
                    WH = self.reconstruct(self.H.detach(), self.W)
                    loss = Beta_divergence(self.fix_neg(WH), V, beta)
                    loss.backward()

                    with torch.no_grad():
                        positive_comps, H_sum = self.get_W_positive(WH, beta, H_sum)
                        _mu_update(self.W, positive_comps, gamma, l1_reg, l2_reg, W_reg_index, 1)
                    W_sum = None

                if self.H.requires_grad:
                    self.zero_grad()
                    WH = self.reconstruct(self.H, self.W.detach())
                    loss = Beta_divergence(self.fix_neg(WH), V, beta)
                    loss.backward()

                    with torch.no_grad():
                        positive_comps, W_sum = self.get_H_positive(WH, beta, W_sum)
                        _mu_update(self.H, positive_comps, gamma, l1_reg, l2_reg, H_reg_index, 0)

                    H_sum = None


                loss = loss.div_(loss_scale).item()

                pbar.set_postfix(loss=loss)
                # pbar.set_description('Beta loss=%.4f' % error)
                pbar.update()

                if not n_iter:
                    loss_init = loss
                elif (previous_loss - loss) / loss_init < tol:
                    break
                previous_loss = loss

        return n_iter

    def fit_transform(self, *args, **kwargs):
        n_iter = self.fit(*args, **kwargs)
        return n_iter, self.forward()

    @torch.no_grad()
    def renorm(self, unit_norm='W'):
        W2 = self.W * self.W
        H2 = self.H * self.H
        sum_dims = list(range(W2.dim()))
        sum_dims.remove(1)
        W_norm = W2.sum(sum_dims, keepdim=True)
        sum_dims = list(range(1, H2.dim()))
        H_norm = H2.sum(sum_dims, keepdim=True)
        scaler = W_norm.squeeze() / H_norm.squeeze()

        if unit_norm == 'W':
            self.W /= W_norm
            scaler = W_norm.squeeze() / H_norm.squeeze()
            slicer = (slice(None),) + (None,) * (self.H.dim() - 1)
            self.H *= scaler[slicer]
        elif unit_norm == 'H':
            self.H /= H_norm
            slicer = (slice(None),) + (None,) * (self.W.dim() - 2)
            self.W /= scaler[slicer]
        else:
            raise ValueError("Input type isn't valid!")

class NMF(_NMF):

    def __init__(self, Vshape, rank=None):
        self.K, self.M = Vshape
        if not rank:
            rank = self.K

        super().__init__((self.K, rank), (rank, self.M), rank)

    def reconstruct(self, H, W):
        return W @ H

    def get_W_positive(self, WH, beta, H_sum):
        H = self.H
        if beta == 1:
            if H_sum is None:
                H_sum = H.sum(1)
            denominator = H_sum[None, :]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WHHt = WH @ H.t()
            denominator = WHHt

        return denominator, H_sum

    def get_H_positive(self, WH, beta, W_sum):
        W = self.W
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

    def sort(self):
        _, maxidx = self.W.data.max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        self.H.data = self.H.data[idx]


class NMFD(_NMF):
    def __init__(self, Vshape, T=1, rank=None):
        self.K, self.M = Vshape
        if not rank:
            rank = self.K
        self.pad_size = T - 1
        super().__init__((self.K, rank, T), (rank, self.M - T + 1), rank)

    def reconstruct(self, H, W):
        return F.conv1d(H[None, :], W.flip(2), padding=self.pad_size)[0]

    def get_W_positive(self, WH, beta, H_sum):
        H = self.H
        if beta == 1:
            if H_sum is None:
                H_sum = H.sum(1)
            denominator = H_sum[None, :, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WHHt = F.conv1d(WH[:, None], H[:, None])
            denominator = WHHt

        return denominator, H_sum

    def get_H_positive(self, WH, beta, W_sum):
        W = self.W
        if beta == 1:
            if W_sum is None:
                W_sum = W.sum((0, 2))
            denominator = W_sum[:, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WtWH = F.conv1d(WH[None, :], W.transpose(0, 1))[0]
            denominator = WtWH
        return denominator, W_sum

    def sort(self):
        _, maxidx = self.W.data.sum(2).max(0)
        _, idx = maxidx.sort()
        self.W.data = self.W.data[:, idx]
        self.H.data = self.H.data[idx]


class NMF2D(_NMF):
    def __init__(self, Vshape, win=1, rank=None):
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
        super().__init__((self.channel, rank, F, T), (rank, self.K - F + 1, self.M - T + 1), rank)

    def reconstruct(self, H, W):
        out = F.conv2d(H[None, ...], W.flip((2, 3)), padding=self.pad_size)[0]
        if self.channel == 1:
            return out[0]
        return out

    def get_W_positive(self, WH, beta, H_sum):
        H = self.H
        if beta == 1:
            if H_sum is None:
                H_sum = H.sum((1, 2))
            denominator = H_sum[None, :, None, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WH = WH.view(self.channel, 1, self.K, self.M)
            WHHt = F.conv2d(WH, H[:, None])
            denominator = WHHt

        return denominator, H_sum

    def get_H_positive(self, WH, beta, W_sum):
        W = self.W
        if beta == 1:
            if W_sum is None:
                W_sum = W.sum((0, 2, 3))
            denominator = W_sum[:, None, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WH = WH.view(1, self.channel, self.K, self.M)
            WtWH = F.conv2d(WH, W.transpose(0, 1))[0]
            denominator = WtWH
        return denominator, W_sum

    def sort(self):
        raise NotImplementedError


class NMF3D(_NMF):
    def __init__(self, Vshape: tuple, rank: int = None, win=1):
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
        super().__init__((self.channel, rank, T, H, W), (rank, self.N - T + 1, self.K - H + 1, self.M - W + 1), rank)

    def reconstruct(self, H, W):
        out = F.conv3d(H[None, ...], W.flip((2, 3, 4)), padding=self.pad_size)[0]
        if self.channel == 1:
            return out[0]
        return out

    def get_W_positive(self, WH, beta, H_sum):
        H = self.H
        if beta == 1:
            if H_sum is None:
                H_sum = H.sum((1, 2, 3))
            denominator = H_sum[None, :, None, None, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WH = WH.view(self.channel, 1, self.N, self.K, self.M)
            WHHt = F.conv3d(WH, H[:, None])
            denominator = WHHt

        return denominator, H_sum

    def get_H_positive(self, WH, beta, W_sum):
        W = self.W
        if beta == 1:
            if W_sum is None:
                W_sum = W.sum((0, 2, 3, 4))
            denominator = W_sum[:, None, None, None]
        else:
            if beta != 2:
                WH = WH.pow(beta - 1)
            WH = WH.view(1, self.channel, self.N, self.K, self.M)
            WtWH = F.conv3d(WH, W.transpose(0, 1))[0]
            denominator = WtWH
        return denominator, W_sum

    def sort(self):
        raise NotImplementedError
