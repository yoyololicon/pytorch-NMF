import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display, feature

from torchnmf import NMF, NMFD, PLCA, SIPLCA, SIPLCA2

if __name__ == '__main__':
    y, sr = librosa.load(librosa.util.example_audio_file())
    y = torch.from_numpy(y)
    windowsize = 2048
    S = torch.stft(y, windowsize, window=torch.hann_window(windowsize)).pow(2).sum(2).sqrt()
    #S = feature.melspectrogram(y, sr, n_fft=windowsize, power=2) ** 0.5
    #S[S == 0] = 1e-8
    S = torch.FloatTensor(S)
    R = 3
    T = 400
    F = S.shape[0] - 1

    net = NMFD(S.shape, T=T, rank=R).cuda()
    _, V, *_ = net.fit_transform(S.cuda(), beta=1, tol=1e-5, verbose=True, max_iter=500, alpha=0.01, l1_ratio=0.1,
                                 H_reg_control=1, W_reg_control=0)
    net.sort()
    net.renorm('W')
    W, H = net.W.detach().cpu().numpy(), net.H.detach().cpu().numpy()
    V = V.detach().cpu().numpy()
    #V = net(net.H[1:2], net.W[:, 1:2, :]).detach().cpu().numpy()

    if len(W.shape) < 3:
        W = W.reshape(*W.shape, 1)

    plt.figure(figsize=(10, 8))
    for i in range(R):
        plt.subplot(3, R, i + 1)
        display.specshow(librosa.amplitude_to_db(W[:, i], ref=np.max), y_axis='log')
        plt.title('Template ' + str(i + 1))
    plt.subplot(3, 1, 2)
    display.specshow(librosa.amplitude_to_db(H, ref=np.max), x_axis='time')
    plt.colorbar()
    plt.title('Activations')
    plt.subplot(3, 1, 3)
    display.specshow(librosa.amplitude_to_db(V, ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed spectrogram')
    plt.tight_layout()
    plt.show()
