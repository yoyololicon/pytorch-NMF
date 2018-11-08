import numpy as np
import torch
import torch.nn.functional as F

def return_torch(x):
    if type(x) == np.ndarray:
        return torch.from_numpy(x).float()
    elif type(x) == torch.Tensor:
        return x
    else:
        raise TypeError