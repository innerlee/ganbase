import numpy as np
import sklearn.preprocessing as prep
import torch
from torch.autograd import Variable
from .latent import *


class TorusLatent(Latent):
    """
    T^n in 2n-dim space

    n: intrisic dim
    confine: hard (geometry without backprop) | soft (geometry with backprop) | regularize (use regularizations)
    note: the three confines are the same for torus.
    """

    def __init__(self, n, confine='hard'):
        super().__init__()
        #region yapf: disable
        self.intrinsic_dim = n
        self.embed_in_dim  = n
        self.embed_out_dim = 2 * n
        self.R             = np.sqrt(self.intrinsic_dim)
        self.confine       = confine
        self.outactivation = 'none'
        if n == 1:
            self.draw_dim = 2
        elif n == 2:
            self.draw_dim = 3
        else:
            self.draw_dim = n
        #endregion yapf: enable

    def sample(self, bs):
        """
        random points at T^n embeded in R^{2n}

        returns np array
        """
        ang = np.random.uniform(-np.pi, np.pi, (bs, self.intrinsic_dim))

        return torch.from_numpy((np.concatenate(
            (np.cos(ang), np.sin(ang)), axis=1)).astype('float32'))

    def embed(self, z):
        """
        from R^{n} to T^n in R^{2n}

        z: a variable
        returns a variable
        """
        assert isinstance(z, Variable)
        assert z.size(1) == self.embed_in_dim

        emb = torch.cat((torch.cos(z), torch.sin(z)), dim=1)
        assert np.abs(emb.data[0, :].norm() / self.R - 1) < 1e-4

        return emb

    def neighbor(self, z, sigma, multiple=1):
        """
        sample one embeded neighbor given points (variable) before embed

        z: a variable
        returns an embeded variable
        """
        assert isinstance(z, Variable)
        z = z.repeat(multiple, 1)
        noise = Variable(sigma * torch.randn(z.size())).cuda(non_blocking=True)
        z = z + noise
        emb = torch.cat((torch.cos(z), torch.sin(z)), dim=1)
        assert torch.abs(emb[0, :].norm() / self.R - 1).data[0] < 1e-5

        return emb

    def embed_draw(self, z):
        """
        from R^{n} to T^n in R^{2n}

        z: a variable
        returns a variable
        """
        assert z.size(1) == self.embed_in_dim

        if self.intrinsic_dim == 1:
            emb = torch.cat((torch.cos(z.data), torch.sin(z.data)), dim=1)
        elif self.intrinsic_dim == 2:
            x = z.data[:, [0]]
            y = z.data[:, [1]]
            emb = torch.cat(
                (torch.cos(x) * (3 + torch.cos(y)),
                 torch.sin(x) * (3 + torch.cos(y)), torch.sin(y)),
                dim=1)
        else:
            emb = torch.fmod(z.data, 2 * np.pi)

        return emb

    def embed_to_draw(self, z):
        """
        embeded points to drawable points

        input: variable
        output: tensor
        """
        if self.intrinsic_dim == 1:
            emb = z.data
        elif self.intrinsic_dim == 2:
            xcos = z.data[:, [0]]
            ycos = z.data[:, [1]]
            xsin = z.data[:, [2]]
            ysin = z.data[:, [3]]
            emb = torch.cat(
                (xcos * (3 + ycos), xsin * (3 + ycos), ysin), dim=1)
        else:
            emb = torch.fmod(z.data, 2 * np.pi)

        return emb
