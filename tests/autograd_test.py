import torch
import librosa
from torchaudio import load
import numpy as np
import matplotlib.pyplot as plt
from librosa import display

from torchnmf.models import NMF, NMFD
from time import time

torch.set_flush_denormal(True)
# torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == '__main__':
    #y, sr = load('/media/ycy/Shared/Datasets/bach10/01-AchGottundHerr/01-AchGottundHerr.wav', normalization=True)
    duration = 60
    y, sr = load('/media/ycy/Shared/Datasets/MAPS/ENSTDkCl/MUS/MAPS_MUS-alb_se2_ENSTDkCl.wav', normalization=True)
    y = y.mean(1)[:duration * sr]
    windowsize = 4096
    S = torch.stft(y, windowsize, window=torch.hann_window(windowsize)).pow(2).sum(2).sqrt()
    S[S == 0] = 1e-8
    R = 88
    T = 5
    max_iter = 200

    S = S.cuda()
    net = NMFD(S.shape, T, n_components=R).cuda()
    #net = NMF(S.shape, n_components=R, max_iter=max_iter, verbose=True, beta_loss=2).cuda()

    niter, V = net.fit_transform(S, verbose=True, beta_loss=1.5, max_iter=max_iter, alpha=0.5, l1_ratio=0.2)
    net.sort()
    W = net.W
    H = net.H

    plt.subplot(3, 1, 1)
    display.specshow(librosa.amplitude_to_db(W.detach().cpu().numpy().mean(2), ref=np.max), y_axis='log', sr=sr)
    plt.title('Template ')
    plt.subplot(3, 1, 2)
    display.specshow(H.detach().cpu().numpy(), x_axis='time', hop_length=1024, sr=sr)
    plt.colorbar()
    plt.title('Activations')
    plt.subplot(3, 1, 3)
    display.specshow(librosa.amplitude_to_db(V.detach().cpu().numpy(), ref=np.max), y_axis='log', x_axis='time', sr=sr, hop_length=1024)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed spectrogram')
    plt.tight_layout()
    plt.show()
