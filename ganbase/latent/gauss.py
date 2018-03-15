import numpy as np
import sklearn.preprocessing as prep
import torch
from torch.autograd import Variable
from .latent import *


class GaussLatent(Latent):
    """
    gauss random space
    """

    def __init__(self, dim, bs):
        super().__init__()
        self.dim = dim
        self.bs = bs

    def sample(self, n):
        """
        random points at sphere S^n embeded in R^{n+1}

        returns np array
        """
        return torch.from_numpy(np.random.randn(n, self.dim).astype('float32'))
