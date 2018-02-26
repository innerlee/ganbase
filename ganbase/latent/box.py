import numpy as np
import sklearn.preprocessing as prep
import torch
from torch.autograd import Variable
from .latent import *


class BoxLatent(Latent):
    """
    [-1, 1]^n in mapped to half circles in 2n-dim space

    n: intrisic dim
    confine: hard (geometry without backprop) | soft (geometry with backprop) | regularize (use regularizations)
    note: regularize not supported for box
    """

    def __init__(self, n, confine='hard'):
        super().__init__()
        #region yapf: disable
        self.intrinsic_dim = n
        self.embed_in_dim  = n
        self.embed_out_dim = 2 * n
        self.draw_dim      = n
        self.R             = np.sqrt(self.intrinsic_dim)
        self.confine       = confine
        self.outactivation = 'tanh'
        self.eps           = 1e-4
        #endregion yapf: enable

    def sample(self, bs):
        """
        random points at box [-1, 1]^n embeded in R^{2n}

        returns np array
        """
        left = np.random.uniform(-1, 1, (bs, self.intrinsic_dim))
        right = np.sqrt(1 - left**2)

        return np.concatenate((left, right), axis=1)

    def embed(self, z):
        """
        from R^{n} to [-1, 1]^n in R^{2n}

        z: a variable
        returns a variable
        """
        assert isinstance(z, Variable)
        z = z.clamp(-1 + self.eps, 1 - self.eps)

        if self.confine == 'soft':
            z_right = torch.sqrt(1 - z**2)
        elif self.confine == 'hard':
            z_right = Variable(torch.sqrt(1 - z.data**2))
        elif self.confine == 'regularize':
            raise ValueError('box latent do not support regularize confine')
        else:
            raise ValueError('latent confine not supported')

        emb = torch.cat((z, z_right), dim=1)
        assert torch.abs(emb[0, :].norm() / self.R - 1).data[0] < 1e-5

        return emb

    def neighbor(self, z, sigma, multiple=1):
        """
        sample one embeded neighbor given points (variable) before embed

        z: a variable
        returns an embeded variable
        """
        assert isinstance(z, Variable)
        z = z.repeat(multiple, 1)
        noise = Variable(sigma * torch.randn(z.size())).cuda(async=True)
        z = torch.clamp(z + noise, -1, 1)
        z_right = Variable(torch.sqrt(1 - z.data**2))
        emb = torch.cat((z, z_right), dim=1)
        assert torch.abs(emb[0, :].norm() / self.R - 1).data[0] < 1e-5

        return emb

    def embed_draw(self, z):
        """
        from R^{n} to [-1, 1]^n in R^{2n}

        z: a variable
        returns a variable
        """
        return z.data.clamp(-1, 1)

    def embed_to_draw(self, z):
        """
        embeded points to drawable points

        input: variable
        output: tensor
        """
        return z.data[:, :(z.size(1) // 2)]
