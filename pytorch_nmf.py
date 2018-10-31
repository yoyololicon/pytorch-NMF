import torch
from torch.nn import functional as F


def generalized_KL_divergence(V, V_tilde):
    return torch.sum(V * torch.log(V / V_tilde) - V + V_tilde)


def NMF(S, R, init_W=None, init_H=None, fix_W=False, fix_H=False, n_iter=50):
    """

    :param S: magnitude spectrogram
    :param R: number of templates
    :param init_W: initial weight
    :param init_H: initial activations
    :param fix_W:
    :param fix_H:
    :param n_iter:
    :return : W, H
    """

    K, M = S.shape

    if init_W is None:
        W = torch.rand(K, R)
    else:
        W = init_W.clone()
    if init_H is None:
        H = torch.rand(R, M)
    else:
        H = init_H.clone()

    for i in range(n_iter):
        V = W @ H
        loss = generalized_KL_divergence(S, V)
        print("Iter", "%d" % (i + 1), ", KL:" "%.4f" % loss)

        SonV = S / V
        # update W
        if not fix_W:
            W *= SonV @ H.t() / H.t().sum(0)

        # update H
        if not fix_H:
            H *= W.t() @ SonV / W.t().sum(1, keepdim=True)

    return W, H


def NMFD(S, R, T, init_W=None, init_H=None, fix_W=False, fix_H=False, n_iter=50):
    """

    :param S: magnitude spectrogram
    :param R: number of templates
    :param T: template length (number of frames)
    :param init_W: initial weight
    :param init_H: initial activations
    :param fix_W:
    :param fix_H:
    :param n_iter:
    :return: W, H
    """

    K, M = S.shape

    if init_W is None:
        W = torch.rand(K, R, T)
    else:
        W = init_W.clone()
    if init_H is None:
        H = torch.rand(R, M)
    else:
        H = init_H.clone()

    for i in range(n_iter):
        V = F.conv1d(H[None, :], W.flip(2), padding=T - 1)[0, :, :M]

        loss = generalized_KL_divergence(S, V)
        print("Iter", "%d" % (i + 1), ", KL:" "%.4f" % loss)

        SonV = S / V
        # update W
        if not fix_W:
            expand_H = torch.stack([F.pad(H[:, :M-j], (j, 0)) for j in range(T)], dim=2)
            upper = (SonV @ expand_H).transpose(0, 1)
            lower = expand_H.transpose(0, 1).sum(0)
            W *= upper / lower

        # update H
        if not fix_H:
            expand_SonV = torch.stack([F.pad(SonV[:, j:], (0, j)) for j in range(T)], dim=0)
            upper = W.transpose(0, 2) @ expand_SonV  # (T, R, M)
            lower = W.transpose(0, 1).sum(1, keepdim=True)
            H *= torch.mean(upper.permute(1, 2, 0) / lower, 2)

    return W, H


if __name__ == '__main__':
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    from librosa import display

    #torch.set_default_tensor_type(torch.DoubleTensor)
    y, sr = librosa.load(librosa.util.example_audio_file())
    S = np.abs(librosa.stft(y))
    S = torch.from_numpy(S)
    R = 8
    T = 5

    comps, acts = NMFD(S, R=R, T=T, n_iter=150)
    V = F.conv1d(acts[None, :], comps.flip(2), padding=T - 1)[0]
    comps = comps.numpy()
    acts = acts.numpy()
    display.specshow(np.log1p(comps[:, 0, :]), y_axis='log')
    plt.show()
    plt.imshow(np.log1p(acts), aspect='auto', origin='lower')
    plt.show()


    display.specshow(np.log1p(V.numpy()), y_axis='log')
    plt.show()
