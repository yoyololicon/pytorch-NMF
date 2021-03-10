import pytest
import torch
import numpy as np
from torch import nn

from torchnmf.plca import *


@pytest.mark.parametrize('rank', [8, None])
@pytest.mark.parametrize('W', [(50, 8), torch.rand(50, 8), None])
@pytest.mark.parametrize('H', [(100, 8), torch.rand(100, 8), None])
@pytest.mark.parametrize('Z', [torch.rand(8), None])
def test_base_valid_construct(rank, W, H, Z):
    if (rank is None) and (H is None) and (W is None) and (Z is None):
        return
    m = BaseComponent(rank, W, H, Z)
    if H is None:
        assert m.H is None
    else:
        tmp = list(range(m.H.ndim))
        tmp.remove(1)
        tmp = m.H.sum(tmp)
        assert torch.allclose(tmp, torch.ones_like(tmp))
    if W is None:
        assert m.W is None
    else:
        tmp = list(range(m.W.ndim))
        tmp.remove(1)
        tmp = m.W.sum(tmp)
        assert torch.allclose(tmp, torch.ones_like(tmp))
    if Z is None and not rank:
        assert m.Z is None
    else:
        tmp = m.Z.sum()
        assert torch.allclose(tmp, torch.ones_like(tmp))


@pytest.mark.parametrize('rank, W, H, Z',
                         [(None, None, None, None),
                          (7, (50, 8), (100, 10), None),
                          (None, torch.rand(50, 8), (100, 10), torch.rand(7)),
                          (None, torch.randn(50, 8), (100, 8), torch.rand(8)),
                          (None, torch.rand(50, 8), (100, 8), torch.randn(8)),
                          (None, (50, 8), torch.rand(100, 10), torch.rand(10)),
                          (8, (50, 8), torch.randn(100, 8), None),
                          (None, torch.rand(50, 8), torch.rand(
                              100, 10), torch.rand(7)),
                          (None, torch.randn(50, 8),
                           torch.rand(100, 8), torch.rand(8)),
                          (None, torch.rand(50, 8), torch.randn(
                              100, 8), torch.rand(8)),
                          (None, torch.rand(50, 8), torch.rand(
                              100, 8), torch.randn(8)),
                          (None, torch.randn(50, 8), torch.rand(
                              100, 8), torch.randn(8)),
                          (None, torch.randn(50, 8),
                           torch.randn(100, 8), torch.rand(8)),
                          (None, torch.rand(50, 8), torch.randn(
                              100, 8), torch.randn(8)),
                          (None, torch.randn(50, 8), torch.randn(100, 8), torch.randn(8)), ]
                         )
def test_base_invalid_construct(rank, W, H, Z):
    try:
        m = BaseComponent(rank, W, H, Z)
    except:
        assert True
    else:
        assert False, "Should not reach here"


def test_plca_valid_construct():
    m = PLCA((100, 50))
    y = m()
    assert y.shape == (100, 50)
    assert torch.allclose(y.sum(), torch.ones(1))


@pytest.mark.parametrize('Vshape', [(100, 50, 50), (100,)])
def test_plca_invalid_construct(Vshape):
    try:
        m = PLCA(Vshape)
    except:
        assert True
    else:
        assert False, "Should not reach here"


def test_siplca_valid_construct():
    m = SIPLCA((100, 50, 100))
    y = m()
    assert y.shape == (100, 50, 100)
    assert torch.allclose(y.sum(), torch.ones(1))


@pytest.mark.parametrize('Vshape', [(100, 50), (100,), (100, 50) * 2])
def test_siplca_invalid_construct(Vshape):
    try:
        m = SIPLCA(Vshape)
    except:
        assert True
    else:
        assert False, "Should not reach here"


def test_siplca2_valid_construct():
    m = SIPLCA2((8, 32, 100, 100), 16)
    y = m()
    assert y.shape == (8, 32, 100, 100)
    assert torch.allclose(y.sum(), torch.ones(1))


@pytest.mark.parametrize('Vshape', [(100, 50), (100,), (100, 50) * 6])
def test_siplca2_invalid_construct(Vshape):
    try:
        m = SIPLCA2(Vshape)
    except:
        assert True
    else:
        assert False, "Should not reach here"


def test_siplca3_valid_construct():
    m = SIPLCA3((8, 50, 100, 100, 100), 8)
    y = m()
    assert y.shape == (8, 50, 100, 100, 100)
    assert torch.allclose(y.sum(), torch.ones(1))


@pytest.mark.parametrize('Vshape', [(100, 50), (100,), (100, 50) * 4])
def test_siplca3_invalid_construct(Vshape):
    try:
        m = SIPLCA3(Vshape)
    except:
        assert True
    else:
        assert False, "Should not reach here"


@pytest.mark.parametrize('tol', [0, 1e-4])
@pytest.mark.parametrize('verbose', [True, False])
@pytest.mark.parametrize('W_alpha', [1, 0.999])
@pytest.mark.parametrize('Z_alpha', [1, 0.999])
@pytest.mark.parametrize('H_alpha', [1, 0.999])
@pytest.mark.parametrize('trainable_Z', [True, False])
@pytest.mark.parametrize('trainable_W', [True, False])
def test_fit(tol,
             verbose,
             W_alpha,
             H_alpha,
             Z_alpha,
             trainable_Z,
             trainable_W):
    max_iter = 100
    V = torch.rand(100, 50)
    m = PLCA(None, 8, H=torch.rand(100, 8), W=torch.rand(50, 8), Z=torch.ones(8) /
             8, trainable_Z=trainable_Z, trainable_W=trainable_W)
    n_iter, norm = m.fit(V, tol, max_iter, verbose, W_alpha, H_alpha, Z_alpha)
    assert n_iter <= max_iter
    y = m(norm=norm)
