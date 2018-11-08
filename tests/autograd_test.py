import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display

from torchnmf.models import NMF as torchNMF, NMFD
from time import time

#torch.set_flush_denormal(True)
#torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == '__main__':
    y, sr = librosa.load(
        '/media/ycy/Shared/Datasets/DSD100subset/Sources/Test/005 - Angela Thomas Wade - Milk Cow Blues/drums.wav')
    y = torch.from_numpy(y)
    windowsize = 2048
    S = torch.stft(y, windowsize, window=torch.hann_window(windowsize)).pow(2).sum(2).sqrt()
    R = 4
    T = 8
    max_iter = 1000

    S = S.cuda()
    start = time()
    net = NMFD(S.shape, T, n_components=R, max_iter=max_iter, verbose=True, beta_loss=1, tol=1e-5).cuda()
    niter, V = net.fit_transform(S.cuda())
    print(niter / (time() - start))
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
