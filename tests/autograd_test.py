import torch
import librosa
from torchaudio import load
import numpy as np
import matplotlib.pyplot as plt
from librosa import display
from scipy.io import loadmat
from torchnmf.plca import PLCA, SIPLCA, SIPLCA2, SIPLCA3
from time import time

torch.set_flush_denormal(True)


def read_bach10_F0s(F0):
    f = np.round(loadmat(F0)['GTF0s'] - 21).astype(int)
    index = np.where(f >= 0)
    pianoroll = np.zeros((88, f.shape[1]))
    for i, frame in zip(index[0], index[1]):
        pianoroll[f[i, frame], frame] = 1
    return pianoroll


if __name__ == '__main__':
    y, sr = librosa.load('/media/ycy/Shared/Datasets/bach10/01-AchGottundHerr/01-AchGottundHerr.wav', sr=None)
    H = read_bach10_F0s('/media/ycy/Shared/Datasets/bach10/01-AchGottundHerr/01-AchGottundHerr-GTF0s.mat').astype(
        np.float32)

    S = librosa.feature.melspectrogram(y, sr, norm=None).astype(np.float32)
    S[S == 0] = 1e-8
    S = np.stack((S, S), 0)
    S = torch.tensor(S)
    # H = torch.tensor(H)
    # H[H == 0] = 1e-8
    R = 1
    win = (30, 5)
    max_iter = 200

    net = SIPLCA2(S.shape, rank=R, win=win, uniform=True).cuda()
    # net = NMF(S.shape, n_components=R, max_iter=max_iter, verbose=True, beta_loss=2).cuda()

    # W = torch.exp(-torch.arange(64.)).view(1, 1, 64, 1)
    # W /= W.sum()

    niter, V = net.fit_transform(S.cuda(), verbose=True, max_iter=max_iter, H_alpha=1.0001)
    #net.sort()
    W = net.W.detach().cpu().numpy().reshape(win[0], -1)
    H = net.H.detach().cpu().numpy()[0]

    plt.subplot(3, 1, 1)
    display.specshow(librosa.amplitude_to_db(W, ref=np.max), y_axis='mel', sr=sr)
    plt.title('Template ')
    plt.subplot(3, 1, 2)
    display.specshow(H, x_axis='time', hop_length=512, sr=sr)
    plt.colorbar()
    plt.title('Activations')
    plt.subplot(3, 1, 3)
    display.specshow(librosa.amplitude_to_db(V.detach().cpu().numpy(), ref=np.max), y_axis='mel', x_axis='time', sr=sr,
                     hop_length=512)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed spectrogram')
    plt.tight_layout()
    plt.show()
