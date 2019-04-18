import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display, feature

from torchnmf import NMF, NMFD, PLCA, SIPLCA, SIPLCA2

if __name__ == '__main__':
    y, sr = librosa.load(librosa.util.example_audio_file())
    #y = torch.from_numpy(y)
    windowsize = 2048
    #S = torch.stft(y, windowsize, window=torch.hann_window(windowsize)).pow(2).sum(2).sqrt()
    S = feature.melspectrogram(y, sr, n_fft=windowsize)
    S[S == 0] = 1e-8
    S = torch.FloatTensor(S)
    R = 8
    T = 1
    F = S.shape[0] - 1

    net = SIPLCA2(S.shape, win=(F, T), n_components=R).cuda()
    _, V = net.fit_transform(S.cuda(), verbose=True, max_iter=100)
    net.sort()
    W, H = net.W.detach().cpu().numpy(), net.H.detach().cpu().numpy()
    V = V.detach().cpu().numpy()

    if len(W.shape) < 3:
        W = W.reshape(*W.shape, 1)

    plt.figure(figsize=(10, 8))
    for i in range(R):
        plt.subplot(3, R, i + 1)
        display.specshow(librosa.power_to_db(W[:, i], ref=np.max), y_axis='mel')
        plt.title('Template ' + str(i + 1))
    plt.subplot(3, 1, 2)
    display.specshow(H[0], x_axis='time')
    plt.colorbar()
    plt.title('Activations')
    plt.subplot(3, 1, 3)
    display.specshow(librosa.power_to_db(V, ref=np.max), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed spectrogram')
    plt.tight_layout()
    plt.show()
