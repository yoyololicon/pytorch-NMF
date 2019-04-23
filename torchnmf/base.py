from torch import nn


class Base(nn.Module):

    def __init__(self):
        super().__init__()
        self.fix_neg = nn.Threshold(0., 1e-8)

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def fit_transform(self, *args, **kwargs):
        raise NotImplementedError

    def sort(self):
        raise NotImplementedError
