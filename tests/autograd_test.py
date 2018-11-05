import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display

from torchnmf.models import *

from sklearn.decomposition import NMF
from time import time

if __name__ == '__main__':
    y, sr = librosa.load(
        '/media/ycy/Shared/Datasets/DSD100subset/Sources/Test/005 - Angela Thomas Wade - Milk Cow Blues/drums.wav')
    y = torch.from_numpy(y)
    windowsize = 2048
    S = torch.stft(y, windowsize, window=torch.hann_window(windowsize)).pow(2).sum(2).sqrt()
    R = 4
    T = 10
    max_iter = 200

    model = NMF(R, 'random', solver='mu', max_iter=max_iter, verbose=True, beta_loss=0)
    start = time()
    W = model.fit_transform(S.numpy())
    print(model.reconstruction_err_, model.n_iter_ / (time() - start))
    H = model.components_
    V = W @ H

    plt.subplot(3, 1, 1)
    display.specshow(librosa.amplitude_to_db(W, ref=np.max), y_axis='log')
    plt.title('Template ')
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


    S = S.cuda()
    net = NMF_Beta(S.shape, R).cuda()

    a = time()
    for i in range(max_iter):
        net.zero_grad()
        V, loss = net(S)
        loss_tmp = loss.item()
        loss.backward()
        W = net.update_W()

        net.zero_grad()
        V, loss = net(S)
        err = loss.item()
        print(i, (err + loss_tmp) / 2)
        loss.backward()
        H = net.update_H()

    print(err, max_iter / (time() - a))
    plt.subplot(3, 1, 1)
    display.specshow(librosa.amplitude_to_db(W.detach().cpu().numpy()[:, :], ref=np.max), y_axis='log')
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
