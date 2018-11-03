import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display

from torchnmf import NMF, NMFD
from torchnmf.metrics import KL_divergence, IS_divergence

if __name__ == '__main__':
    y, sr = librosa.load(librosa.util.example_audio_file())
    y = torch.from_numpy(y)
    windowsize = 2048
    S = torch.stft(y, windowsize, window=torch.hann_window(windowsize)).pow(2).sum(2).sqrt()
    R = 8
    K, M = S.shape

    W = torch.nn.Parameter(torch.rand(K, R))
    H = torch.nn.Parameter(torch.rand(R, M))

    loss_fn = torch.nn.MSELoss(reduction='sum')

    for i in range(50):

        if W.grad is not None:
            W.grad.data.zero_()
        V_tilde = W @ H

        loss = loss_fn(V_tilde, S) / 2
        print(i, loss.item())
        loss.backward()

        positive = W @ H @ H.t()
        negative = positive - W.grad.data
        W.data *= negative / positive

        if H.grad is not None:
            H.grad.data.zero_()
        V_tilde = W @ H
        loss = loss_fn(V_tilde, S) / 2
        print(i, loss.item())
        loss.backward()

        positive = W.t() @ W @ H
        negative = positive - H.grad.data
        H.data *= negative / positive

    plt.subplot(3, 1, 1)
    display.specshow(librosa.amplitude_to_db(W.detach().numpy(), ref=np.max), y_axis='log')
    plt.title('Template ')
    plt.subplot(3, 1, 2)
    display.specshow(H.detach().numpy(), x_axis='time')
    plt.colorbar()
    plt.title('Activations')
    plt.subplot(3, 1, 3)
    display.specshow(librosa.amplitude_to_db(V_tilde.detach().numpy(), ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed spectrogram')
    plt.tight_layout()
    plt.show()

    W = torch.nn.Parameter(torch.rand(K, R))
    H = torch.nn.Parameter(torch.rand(R, M))

    loss_fn = KL_divergence

    for i in range(50):

        if W.grad is not None:
            W.grad.data.zero_()
        V_tilde = W @ H
        loss = loss_fn(S, V_tilde)
        loss.backward()

        positive = H.sum(1, keepdim=True).t()
        negative = positive - W.grad.data
        W.data *= negative / positive

        if H.grad is not None:
            H.grad.data.zero_()
        V_tilde = W @ H
        loss = loss_fn(S, V_tilde)
        print(i, loss.item())
        loss.backward()

        positive = W.sum(0, keepdim=True).t()
        negative = positive - H.grad.data
        H.data *= negative / positive

    plt.subplot(3, 1, 1)
    display.specshow(librosa.amplitude_to_db(W.detach().numpy(), ref=np.max), y_axis='log')
    plt.title('Template ')
    plt.subplot(3, 1, 2)
    display.specshow(H.detach().numpy(), x_axis='time')
    plt.colorbar()
    plt.title('Activations')
    plt.subplot(3, 1, 3)
    display.specshow(librosa.amplitude_to_db(V_tilde.detach().numpy(), ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed spectrogram')
    plt.tight_layout()
    plt.show()

    loss_fn = IS_divergence

    for i in range(50):

        if W.grad is not None:
            W.grad.data.zero_()
        V_tilde = W @ H
        loss = loss_fn(S, V_tilde)
        loss.backward()

        positive = (1 / V_tilde) @ H.t()
        negative = positive - W.grad.data
        W.data *= negative / positive

        if H.grad is not None:
            H.grad.data.zero_()
        V_tilde = W @ H
        loss = loss_fn(S, V_tilde)
        print(i, loss.item())
        loss.backward()

        positive = W.t() @ (1 / V_tilde)
        negative = positive - H.grad.data
        H.data *= negative / positive

    plt.subplot(3, 1, 1)
    display.specshow(librosa.amplitude_to_db(W.detach().numpy(), ref=np.max), y_axis='log')
    plt.title('Template ')
    plt.subplot(3, 1, 2)
    display.specshow(H.detach().numpy(), x_axis='time')
    plt.colorbar()
    plt.title('Activations')
    plt.subplot(3, 1, 3)
    display.specshow(librosa.amplitude_to_db(V_tilde.detach().numpy(), ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Reconstructed spectrogram')
    plt.tight_layout()
    plt.show()
