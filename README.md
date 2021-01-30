# Non-negative Matrix Fatorization in PyTorch

PyTorch is not only a good deep learning framework, but also a fast tool when it comes to matrix operations and convolutions on large data.
A great example is [PyTorchWavelets](http://github.com/tomrunia/PyTorchWavelets).


In this package I implement NMF, PLCA and their deconvolutional variations in PyTorch based on `torch.nn.Module`, 
so the models can be moved freely among CPU/GPU devices and utilize parallel computation of cuda.

# Modules

## NMF

Basic NMF and NMFD module minimizing beta-divergence using multiplicative update rules.
The multiplier is obtained via `torch.autograd` so the amount of codes is reduced and easy to maintain.

The interface is similar to `sklearn.decomposition.NMF` with some extra options.

* `NMF`: Original NMF algorithm.
* `NMFD`: 1-D deconvolutional NMF algorithm.
* `NMF2D`: 2-D deconvolutional NMF algorithm. 
* `NMF3D`: 3-D deconvolutional NMF algorithm. 

## PLCA (not documented)

Basic PLCA and SIPLCA module using EM algorithm to minimize
KL-divergence between the target distribution P(X) and the estimated
distribution.

* `PLCA`: Original PLCA (Probabilistic Latent Component Analysis)
  algorithm.
* `SIPLCA`: Shift-Invariant PLCA algorithm (similar to NMFD).
* `SIPLCA2`: 2-D deconvolutional SIPLCA algorithm.
* `SIPLCA3`: 3-D deconvolutional SIPLCA algorithm.

---
**NOTE**

This module is currently not documented and still using the old function interface (before version 0.3).
Will be updated and adopt in later version.

---

## Usage

Here is a short example of decompose a spectrogram using deconvolutional NMF:

```python
import torch
import librosa
from torchnmf.nmf import NMFD
from torchnmf.metrics import kl_div

y, sr = librosa.load(librosa.util.example_audio_file())
y = torch.from_numpy(y)
windowsize = 2048
S = torch.stft(y, windowsize, window=torch.hann_window(windowsize)).pow(2).sum(2).sqrt().cuda()
S = S.unsqueeze(0)

R = 8   # number of components
T = 400 # size of convolution window

net = NMFD(S.shape, rank=R, T=T).cuda()
# run extremely fast on gpu
net.fit(S)      # fit to target matrix S
V = net()
print(kl_div(V, S))        # KL divergence to S
```
A more detailed version can be found [here](tests/librosa_example.py). 
See our [documentation](https://pytorch-nmf.readthedocs.io/en/latest/) to find out more usage of this package.

![](tests/librosa_example.png)

## Compare to sklearn

The barchart shows the time cost per iteration with different
beta-divergence. It shows that pytorch-based NMF has a much more constant process time across 
different beta values, which can take advantage when beta is not 0, 1, or 2.
This is because our implementation use the same computational graph regardless which beta-divergence are we minimizing.
It runs even faster when computation is done on GPU. The test is conducted on a
Acer E5 laptop with i5-7200U CPU and GTX 950M GPU.


![](tests/performance.png) 

## Installation

```
pip install torchnmf
```

## Requirements

* PyTorch
* tqdm

## Tips

* If you notice significant slow down when operating on CPU, please flush denormal numbers by `
torch.set_flush_denormal(True)`.


## TODO

- [ ] Support sparse matrix.
- [x] Regularization.
- [ ] NNDSVD initialization.
- [x] 2/3-D deconvolutional module.
- [x] PLCA.
- [x] Documentation.
- [ ] ipynb examples.
- [ ] Refactor PLCA module.