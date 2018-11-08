import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display

from torchnmf.models import NMF, NMFD
from time import time

torch.set_flush_denormal(True)
# torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == '__main__':
    #y, sr = librosa.load('/media/ycy/Shared/Datasets/bach10/01-AchGottundHerr/01-AchGottundHerr.wav')
    y, sr = librosa.load('/media/ycy/Shared/Datasets/MAPS/ENSTDkCl/MUS/MAPS_MUS-alb_se2_ENSTDkCl.wav', duration=60)
    y = torch.from_numpy(y)
    windowsize = 2048
    S = torch.stft(y, windowsize, window=torch.hann_window(windowsize)).pow(2).sum(2).sqrt()
    S[S == 0] = 1e-8
    R = 88
    T = 5
    max_iter = 1000

    S = S.cuda()
    net = NMFD(S.shape, T, n_components=R, max_iter=max_iter, verbose=True, beta_loss=1).cuda()
    #net = NMF(S.shape, n_components=R, max_iter=max_iter, verbose=True, beta_loss=2).cuda()

    start = time()
    niter, V = net.fit_transform(S)
    print(niter / (time() - start))
    net.sort()
    W = net.W
    H = net.H

    plt.subplot(3, 1, 1)
    display.specshow(librosa.amplitude_to_db(W.detach().cpu().numpy()[:, 0, :], ref=np.max), y_axis='log')
    plt.title('Template ')
    plt.subplot(3, 1, 2)
    display.specshow(H.detach().cpu().numpy(), x_axis='time')
    plt.colorbar()
    plt.title('Activations')
    plt.subplot(3, 1, 3)
    display.specshow(librosa.amplitude_to_db(V.detach().cpu().numpy(), ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed spectrogram')
    plt.tight_layout()
    plt.show()
