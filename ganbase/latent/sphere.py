import numpy as np
import sklearn.preprocessing as prep
import torch
from torch.autograd import Variable
from .latent import *


class SphereLatent(Latent):
    """
    S^n in (n+1)-dim space

    n: intrisic dim
    confine: hard (geometry without backprop) | soft (geometry with backprop) | regularize (use regularizations)
    """

    def __init__(self, n, confine='hard'):
        super().__init__()
        #region yapf: disable
        self.intrinsic_dim = n
        self.embed_in_dim  = n + 1
        self.embed_out_dim = n + 1
        self.draw_dim      = n + 1
        self.R             = np.sqrt(self.embed_out_dim)
        self.confine       = confine
        self.eps           = 1e-8
        self.outactivation = 'none'
        self.reg_loss      = None
        #endregion yapf: enable

    def sample(self, bs):
        """
        random points at sphere S^n embeded in R^{n+1}

        returns np array
        """
        return prep.normalize(np.random.randn(bs, self.embed_out_dim)) * self.R

    def embed(self, z):
        """
        from R^{n+1} to S^n in R^{n+1}

        z: a variable
        returns a variable
        """
        assert isinstance(z, Variable)
        assert z.size(1) == self.embed_in_dim

        if self.confine == 'soft':
            z_norm = z.norm(2, dim=1, keepdim=True) + self.eps
        elif self.confine == 'hard':
            z_norm = Variable(z.data.norm(2, dim=1, keepdim=True) + self.eps)
        elif self.confine == 'regularize':
            z_norm = z.norm(2, dim=1, keepdim=True) + self.eps
            z_norm_norm = z_norm / self.R
            assert z_norm_norm.size() == (z.size(0), 1)
            self.reg_loss = z_norm_norm - torch.log(z_norm_norm)
            return z
        else:
            raise ValueError('latent confine not supported')

        emb = z / z_norm * self.R
        assert torch.abs(emb[0, :].norm() / self.R - 1).data[0] < 1e-3

        return emb

    def neighbor(self, z, sigma, multiple=1):
        """
        sample one embeded neighbor given points (variable) before embed

        z: a variable
        multiple: how many neighbors for each point
        returns an embeded variable
        """
        assert isinstance(z, Variable)
        z = z.repeat(multiple, 1)
        z_norm = Variable(z.data.norm(2, dim=1, keepdim=True) + self.eps)
        unitz = z / z_norm
        noise = Variable(sigma * torch.randn(z.size())).cuda(async=True)
        vert = noise - (unitz * noise).sum(dim=1, keepdim=True) * unitz
        z = unitz * self.R + vert
        z_norm = Variable(z.data.norm(2, dim=1, keepdim=True) + self.eps)
        emb = z / z_norm * self.R
        assert torch.abs(emb[0, :].norm() / self.R - 1).data[0] < 1e-5

        return emb

    def embed_draw(self, z):
        """
        from R^{n+1} to S^n in R^{n+1}

        z: a variable
        returns a tensor
        """
        assert z.size(1) == self.embed_in_dim

        z_norm = z.data.norm(2, dim=1, keepdim=True) + self.eps

        emb = z.data / z_norm * self.R
        assert np.abs(emb[0, :].norm() / self.R - 1) < 1e-5

        return emb

    def embed_to_draw(self, z):
        """
        embeded points to drawable points

        input: variable
        output: tensor
        """
        return z.data
