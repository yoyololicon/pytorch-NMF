import torch
import pytest
from torchnmf.metrics import *


@pytest.mark.parametrize('beta', [-1, 0, 0.5, 1, 1.5, 2, 3])
@pytest.mark.parametrize('x, y', [(torch.zeros(100), torch.rand(100)),
                                  (torch.rand(100), torch.rand(100)),
                                  (torch.rand(100), torch.zeros(100)),
                                  (torch.zeros(100), torch.zeros(100))])
def test_beta_value_range(beta, x, y):
    loss = beta_div(x, y, beta)
    assert not torch.any(torch.isnan(loss)), loss.item()
    assert not torch.any(loss < 0), loss.item()


@pytest.mark.parametrize('x', [torch.rand(100)])
def test_sparseness_value_range(x):
    loss = sparseness(x)
    assert not torch.any(torch.isnan(loss)), loss.item()