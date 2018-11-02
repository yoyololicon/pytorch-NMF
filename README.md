# Non-negative Matrix Fatorization in PyTorch

PyTorch is not only a good deep learning framework, but also a fast tool when it comes to matrix operations and convolution.
A great example is [PyTorchWavelets](http://github.com/tomrunia/PyTorchWavelets).
 
In this package I implement basic NMF and NMFD module base on `torch.nn.Module`, so the models can be moved freely
 among CPU/GPU devices.

More features and different methods will be add in the future, and will compare how fast it can against scipy based implementations like [nimfa](https://github.com/marinkaz/nimfa).

# Usage

Here is a short example of decompose a spectrogram.
```python
import torch
import librosa
from torchnmf import NMF
from torchnmf.metrics import KL_divergence

y, sr = librosa.load(librosa.util.example_audio_file())
y = torch.from_numpy(y)
windowsize = 2048
S = torch.stft(y, windowsize, window=torch.hann_window(windowsize)).pow(2).sum(2).sqrt().cuda()

R = 8   # number of components
n_iter = 100    # number of iterations

net = NMF(S.shape, R=R).cuda()    # run extremely faster on gpu
comps, acts = net(S, n_iter)      # fit to target matrix S
V = net.reconstruct(comps, acts)  # the reconstructed matrix, in NMF is equal to comps @ acts
print(KL_divergence(S, V))        # KL divergence to S
```
A more detailed version can be found [here](tests/librosa_example.py), which redo this [example](https://librosa.github.io/librosa/generated/librosa.decompose.decompose.html#librosa.decompose.decompose)
with NMFD.

![](tests/librosa_example.png)

## Install

Using pip:
```
pip install git+http://github.com/yoyololicon/pytorch-NMFs
```

Or clone this repo and do:
```
python setup.py install
```

## Requirements

* Numpy
* Pytorch
