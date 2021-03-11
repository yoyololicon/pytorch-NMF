import pytest
import torch
import numpy as np
from torch import nn

from torchnmf.trainer import *
from torchnmf.nmf import NMF
from torchnmf.metrics import beta_div


@pytest.mark.parametrize('beta', [-1, 0, 0.5, 1, 1.5, 2, 3])
@pytest.mark.parametrize('l1_reg', [0, 1e-3])
@pytest.mark.parametrize('l2_reg', [0, 1e-3])
@pytest.mark.parametrize('orthogonal', [0, 1e-2])
def test_beta_trainer(beta, l1_reg, l2_reg, orthogonal):
    m = nn.Sequential(
        NMF((100, 16), rank=8),
        NMF(W=(32, 16)),
        NMF(W=(50, 32))
    )

    target = torch.rand(100, 50)
    trainer = BetaMu(m.parameters(), beta, l1_reg, l2_reg, orthogonal)

    def closure():
        trainer.zero_grad()
        return target, m(None)

    for _ in range(10):
        trainer.step(closure)
    return


@pytest.mark.parametrize('attr', ['W', 'H'])
def test_sparse_trainer(attr):
    m = NMF((100, 50))

    target = torch.rand(100, 50)
    trainer = SparsityProj([getattr(m, attr)], 0.2)

    def closure():
        trainer.zero_grad()
        output = m(None)
        loss = beta_div(output, target)
        return loss

    for _ in range(10):
        trainer.step(closure)
    return


@pytest.mark.parametrize('beta', [-1, 0, 0.5, 1, 1.5, 2, 3])
@pytest.mark.parametrize('attr', ['W', 'H'])
def test_beta_trainer_grad(beta, attr):
    m1 = NMF((100, 50))
    m2 = NMF((100, 50))
    m2.load_state_dict(m1.state_dict())

    target = torch.rand(100, 50)

    trainer = BetaMu([getattr(m1, attr)], beta)

    def closure():
        trainer.zero_grad()
        return target, m1()
    trainer.step(closure)

    loss = beta_div(m2(), target, beta)
    loss.backward()

    assert torch.allclose(getattr(m1, attr).grad, getattr(m2, attr).grad)
