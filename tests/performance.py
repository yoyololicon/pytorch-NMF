from torchaudio import load
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchnmf.models import NMF as torchNMF

from sklearn.decomposition import NMF
from time import time

# torch.set_flush_denormal(True)
#torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == '__main__':
    y, sr = load(
        '/media/ycy/Shared/Datasets/DSD100subset/Sources/Test/005 - Angela Thomas Wade - Milk Cow Blues/drums.wav',
        normalization=True)
    windowsize = 2048
    S = torch.stft(y.mean(1), windowsize, window=torch.hann_window(windowsize)).pow(2).sum(2).sqrt()
    R = 4
    max_iter = 80

    betas = [0, 0.5, 1, 1.5, 2]
    sk = []
    tch = []
    tchcuda = []

    for b in betas:
        print('beta =', b)
        model = NMF(R, 'random', solver='mu', max_iter=max_iter, beta_loss=b, verbose=True)
        start = time()
        model.fit(S.numpy())
        rate = model.n_iter_ / (time() - start)
        print('sklearn', rate)
        sk.append(rate)

        net = torchNMF(S.shape, n_components=R, max_iter=max_iter, beta_loss=b, verbose=True)
        start = time()
        niter = net.fit(S)
        rate = niter / (time() - start)
        print('torch', rate)
        tch.append(rate)

        S2 = S.cuda()
        net = torchNMF(S.shape, n_components=R, max_iter=max_iter, beta_loss=b, verbose=True).cuda()
        start = time()
        niter = net.fit(S2)
        rate = niter / (time() - start)
        print('torch+cuda', rate)
        tchcuda.append(rate)

    plt.bar(np.array(betas) - 0.1, sk, width=0.1, align='center', label='sklearn')
    plt.bar(betas , tch, width=0.1, align='center', label='torch')
    plt.bar(np.array(betas) + 0.1, tchcuda, width=0.1, align='center', label='torch+cuda')
    plt.xlabel(r'$\beta$')
    plt.ylabel("iterations/s")
    plt.legend()
    plt.xticks(betas, [str(i) for i in betas])
    plt.show()