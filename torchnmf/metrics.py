import torch


def generalized_KL_divergence(V, V_tilde):
    id1, *id2 = V.nonzero().t()
    idx = [id1] + id2
    return torch.sum(V[idx] * torch.log(V[idx] / V_tilde[idx])) + torch.sum(V_tilde - V)
