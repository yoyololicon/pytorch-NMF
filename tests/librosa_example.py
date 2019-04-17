import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display

from torchnmf import NMF, NMFD, PLCA

if __name__ == '__main__':
    y, sr = librosa.load(librosa.util.example_audio_file())
    y = torch.from_numpy(y)
    windowsize = 2048
    S = torch.stft(y, windowsize, window=torch.hann_window(windowsize)).pow(2).sum(2).sqrt().cuda()
    S[S == 0] = 1e-8
    R = 16
    T = 5

    net = PLCA(S.shape, rank=R).cuda()
    _, V = net.fit_transform(S, verbose=True, max_iter=50)
    net.sort()
    W, H = net.W.detach().cpu().numpy(), net.H.detach().cpu().numpy()
    V = V.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    for i in range(1):
        plt.subplot(3, 1, i + 1)
        display.specshow(librosa.amplitude_to_db(W[:, :], ref=np.max), y_axis='log')
        plt.title('Template ' + str(i + 1))
    plt.subplot(3, 1, 2)
    display.specshow(H, x_axis='time')
    plt.colorbar()
    plt.title('Activations')
    plt.subplot(3, 1, 3)
    display.specshow(librosa.amplitude_to_db(V, ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed spectrogram')
    plt.tight_layout()
    plt.show()
