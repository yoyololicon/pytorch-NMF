import pytest
import torch


from torchnmf.nmf import *


@pytest.mark.parametrize('beta', [-1, 0, 0.5, 1, 1.5, 2, 3])
@pytest.mark.parametrize('alpha', [0, 0.1])
@pytest.mark.parametrize('l1_ratio', [0, 0.5, 1.])
def test_fit_sparse_target(beta,
                           alpha,
                           l1_ratio):
    max_iter = 5
    Vshape = (800, 800)

    V = torch.rand(*Vshape)

    indices = torch.nonzero(V > 0.95).T
    V_sparse = torch.sparse_coo_tensor(
        indices, V[indices[0], indices[1]], Vshape)
    V_dense = V_sparse.to_dense()

    dense_model = NMF(Vshape, 16)
    sparse_model = NMF(Vshape, 16)
    #dense_model = torch.jit.script(dense_model)
    #sparse_model = torch.jit.script(sparse_model)
    sparse_model.load_state_dict(dense_model.state_dict())

    n_iter = dense_model.fit(
        V_dense, beta, 0, max_iter, False, alpha, l1_ratio)
    n_iter = sparse_model.fit(
        V_sparse, beta, 0, max_iter, False, alpha, l1_ratio)
    assert torch.allclose(dense_model.W, sparse_model.W), torch.abs(
        dense_model.W - sparse_model.W).max().item()
    assert torch.allclose(dense_model.H, sparse_model.H), torch.abs(
        dense_model.H - sparse_model.H).max().item()


@pytest.mark.parametrize('beta,sW,sH', [(2, 0.3, None),
                                        (2, None, 0.3),
                                        (1.5, 0.4, None),
                                        (1.5, None, 0.4),
                                        (1, 0.1, None),
                                        (1, None, 0.1),
                                        (0.5, 0.5, None),
                                        (0.5, None, 0.5),
                                        (0, 0.5, None),
                                        (0, None, 0.5),
                                        (2.5, 0.1, None),
                                        (2.5, None, 0.1),
                                        # (-0.5, 0.3, None),         # beta < 0 very unstable
                                        #(-0.5, None, 0.3),
                                        ])
def test_sparse_fit_sparse_target(beta,
                                  sW,
                                  sH):

    torch.random.manual_seed(2434)
    max_iter = 1
    Vshape = (800, 800)

    V = torch.rand(*Vshape)

    indices = torch.nonzero(V > 0.95).T
    V_sparse = torch.sparse_coo_tensor(
        indices, V[indices[0], indices[1]], Vshape)
    V_dense = V_sparse.to_dense()

    dense_model = NMF(Vshape, 16)
    sparse_model = NMF(Vshape, 16)
    #dense_model = torch.jit.script(dense_model)
    #sparse_model = torch.jit.script(sparse_model)
    sparse_model.load_state_dict(dense_model.state_dict())

    n_iter = dense_model.sparse_fit(
        V_dense, beta, max_iter, False, sW, sH)
    n_iter = sparse_model.sparse_fit(
        V_sparse, beta, max_iter, False, sW, sH)
    assert torch.allclose(dense_model.W, sparse_model.W, atol=2e-6), torch.abs(
        dense_model.W - sparse_model.W).max().item()
    assert torch.allclose(dense_model.H, sparse_model.H, atol=2e-6), torch.abs(
        dense_model.H - sparse_model.H).max().item()
