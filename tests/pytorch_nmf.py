import torch
from torch.nn import functional as F

from torchnmf import NMF, NMFD
from torchnmf.metrics import generalized_KL_divergence

if __name__ == '__main__':
    import librosa
    from torchaudio import load
    import numpy as np
    import matplotlib.pyplot as plt
    from librosa import display

    # torch.set_default_tensor_type(torch.DoubleTensor)
    y, sr = librosa.load(librosa.util.example_audio_file())
    y = torch.from_numpy(y)
    S = torch.stft(y, 2048, window=torch.hann_window(2048)).pow(2).sum(2).sqrt().cuda()
    # S = torch.from_numpy(S)
    R = 8
    T = 5

    net = NMFD(S.shape, R=R).cuda()
    comps, acts = net(S, 100)
    V = net.reconstruct(comps, acts)
    comps = comps.cpu().numpy()
    acts = acts.cpu().numpy()
    display.specshow(np.log1p(comps[:, 0, :]), y_axis='log')
    plt.show()
    plt.imshow(np.log1p(acts), aspect='auto', origin='lower')
    plt.show()

    print(generalized_KL_divergence(S, V))
    V = V.cpu().numpy()
    display.specshow(np.log1p(V), y_axis='log')
    plt.show()
