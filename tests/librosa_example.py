import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display

from torchnmf import NMF, NMFD
from torchnmf.metrics import KL_divergence

if __name__ == '__main__':
    y, sr = librosa.load(librosa.util.example_audio_file())
    y = torch.from_numpy(y)
    windowsize = 2048
    S = torch.stft(y, windowsize, window=torch.hann_window(windowsize)).pow(2).sum(2).sqrt().cuda()
    R = 8
    T = 5

    net = NMFD(S.shape, R=R, T=T).cuda()
    print("step / KL divergence")
    for i in range(200):
        comps, acts = net(S)
        V = net.reconstruct()
        print(i + 1, KL_divergence(S, V).item())
    comps = comps.cpu().numpy()
    acts = acts.cpu().numpy()
    V = V.cpu().numpy()

    plt.figure(figsize=(10, 8))

    for i in range(R):
        plt.subplot(3, R, i + 1)
        display.specshow(librosa.amplitude_to_db(comps[:, i], ref=np.max), y_axis='log')
        plt.title('Template ' + str(i + 1))
    plt.subplot(3, 1, 2)
    display.specshow(acts, x_axis='time')
    plt.colorbar()
    plt.title('Activations')
    plt.subplot(3, 1, 3)
    display.specshow(librosa.amplitude_to_db(V, ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed spectrogram')
    plt.tight_layout()
    plt.show()
