import numpy as np
import sklearn.preprocessing as prep
import torch
from torch.autograd import Variable
from .latent import *


class GaussLatent(object):
    """
    gauss random space
    """

    def __init__(self, dim):
        self.dim = dim

    def sample(self, bs):
        """
        random points at sphere S^n embeded in R^{n+1}

        returns np array
        """
        return np.random.randn(bs, self.dim)
