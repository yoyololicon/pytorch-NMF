import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from librosa import display

from torchnmf import NMF, NMFD
from torchnmf.metrics import KL_divergence, IS_divergence, Frobenius

if __name__ == '__main__':
    y, sr = librosa.load(librosa.util.example_audio_file())
    y = torch.from_numpy(y)
    S = torch.stft(y, 2048, window=torch.hann_window(2048)).pow(2).sum(2).sqrt().cuda()
    R = 8
    T = 5

    net = NMF(S.shape, R=R).cuda()
    comps, acts = net(S, 100)
    V = net.reconstruct(comps, acts)
    comps = comps.cpu().numpy()
    acts = acts.cpu().numpy()
    display.specshow(np.log1p(comps[:,  :]), y_axis='log')
    plt.show()
    plt.imshow(np.log1p(acts), aspect='auto', origin='lower')
    plt.show()

    print(KL_divergence(S, V).item())
    V = V.cpu().numpy()
    display.specshow(np.log1p(V), y_axis='log')
    plt.show()
