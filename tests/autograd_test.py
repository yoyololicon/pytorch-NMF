import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display

from torchnmf.models import *
from torchnmf.metrics import KL_divergence, IS_divergence, Euclidean

if __name__ == '__main__':
    y, sr = librosa.load(librosa.util.example_audio_file())
    y = torch.from_numpy(y)
    windowsize = 2048
    S = torch.stft(y, windowsize, window=torch.hann_window(windowsize)).pow(2).sum(2).sqrt().cuda()
    R = 16

    net = NMF_KL(S.shape, R).cuda()

    for i in range(500):
        net.zero_grad()
        V, loss = net(S)
        loss_tmp = loss.item()
        loss.backward()
        W = net.update_W()

        net.zero_grad()
        V, loss = net(S)
        print(i, (loss.item() + loss_tmp) / 2)
        loss.backward()
        H = net.update_H()

    plt.subplot(3, 1, 1)
    display.specshow(librosa.amplitude_to_db(W.detach().cpu().numpy(), ref=np.max), y_axis='log')
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
