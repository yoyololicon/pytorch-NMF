import torch
import librosa
from torchaudio import load
import numpy as np
import matplotlib.pyplot as plt
from librosa import display
from scipy.io import loadmat
from torchnmf.models import NMF, NMFD, PLCA
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
    y, sr = load('/media/ycy/Shared/Datasets/bach10/01-AchGottundHerr/01-AchGottundHerr.wav', normalization=True)
    y = y.mean(0)
    H = read_bach10_F0s('/media/ycy/Shared/Datasets/bach10/01-AchGottundHerr/01-AchGottundHerr-GTF0s.mat')

    windowsize = 4096
    S = torch.stft(y, windowsize, hop_length=441, window=torch.hann_window(windowsize)).pow(2).sum(2).sqrt()[:,
        :H.shape[1]]
    S[S == 0] = 1e-8
    H = torch.Tensor(H)
    H[H==0] = 1e-8
    R = 88
    T = 5
    max_iter = 10

    S = S
    net = PLCA(S.shape, rank=R)
    # net = NMF(S.shape, n_components=R, max_iter=max_iter, verbose=True, beta_loss=2).cuda()

    niter, V = net.fit_transform(S, H=H, verbose=True, beta_loss=1, max_iter=max_iter)
    #net.sort()
    W = net.W
    H = net.H

    plt.subplot(3, 1, 1)
    display.specshow(librosa.amplitude_to_db(W.detach().cpu().numpy(), ref=np.max), y_axis='log', sr=sr)
    plt.title('Template ')
    plt.subplot(3, 1, 2)
    display.specshow(H.detach().cpu().numpy(), x_axis='time', hop_length=1024, sr=sr)
    plt.colorbar()
    plt.title('Activations')
    plt.subplot(3, 1, 3)
    display.specshow(librosa.amplitude_to_db(V.detach().cpu().numpy(), ref=np.max), y_axis='log', x_axis='time', sr=sr,
                     hop_length=1024)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed spectrogram')
    plt.tight_layout()
    plt.show()
