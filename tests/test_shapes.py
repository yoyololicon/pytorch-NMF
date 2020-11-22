import pytest
import torch
import numpy as np
from torch import nn

from torchnmf.deep import BaseComponent, NMF

@pytest.mark.parametrize('rank', [8])
@pytest.mark.parametrize('W', [(50, 8), torch.rand(50, 8), None])
@pytest.mark.parametrize('H', [(100, 8), torch.rand(100, 8), None])
def test_base_shape(rank, W, H):
    m = BaseComponent(rank, W, H)


@pytest.mark.parametrize('Vsahpe', [(100, 50), None])
@pytest.mark.parametrize('W', [(50, 8), torch.rand(50, 8)])
@pytest.mark.parametrize('H', [None])
def test_nmf_shape(Vsahpe, W, H):
    m = NMF(Vsahpe, W=W, H=H)