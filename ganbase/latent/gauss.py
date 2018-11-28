import numpy as np
import sklearn.preprocessing as prep
import torch
from torch.autograd import Variable
from .latent import *


class GaussLatent(Latent):
    """
    gauss random space
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        #self.bs = bs

    def sample_gauss(self, n):
        """
        random points at sphere S^n embeded in R^{n+1}

        returns np array
        """
        return torch.randn(n, self.dim)

    def sample_uniform(self, n):
        return 2 * torch.rand(n,self.dim) - 1