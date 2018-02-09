"""
MLP Models
"""
from collections import OrderedDict
import torch.nn as nn
import numpy as np

class View(nn.Module):
    """
    reshape layer
    """
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def build_mlp(nin, nout, width, depth, activation, normalize, outactivation):
    """
    Build MLP net.

    activation: leakyrelu | relu | elu | selu | sigmoid | tanh
    normalize: batch | instance | none
    width: int | tuple | list (with len = depth - 1)
    outactivation: none | tanh | sigmoid
    """

    if isinstance(width, (list, tuple)):
        assert len(width) == depth - 1
    elif isinstance(width, int):
        width = [width] * (depth - 1)
    else:
        raise ValueError('width type not supported')

    width = [nin] + list(width)

    net = OrderedDict()

    for i in range(depth-1):
        net.update([(f'fc{i}', nn.Linear(width[i], width[i+1], bias=True))])

        if normalize == 'batch':
            net.update([(f'fc{i}_batchnorm', nn.BatchNorm1d(width[i+1]))])
        elif normalize == 'instance':
            net.update([(f'fc{i}_instancenorm_pre', View(-1, 1, width[i+1]))])
            net.update([(f'fc{i}_instancenorm', nn.InstanceNorm1d(1, affine=False))])
            net.update([(f'fc{i}_instancenorm_post', View(-1, width[i+1]))])
        elif normalize == 'none':
            pass
        else:
            raise ValueError('normalize not supported')

        if activation == 'leakyrelu':
            net.update([(f'fc{i}_leakyrelu', nn.LeakyReLU(0.2, inplace=True))])
        elif activation == 'relu':
            net.update([(f'fc{i}_relu', nn.ReLU(True))])
        elif activation == 'elu':
            net.update([(f'fc{i}_elu', nn.ELU(inplace=True))])
        elif activation == 'selu':
            net.update([(f'fc{i}_selu', nn.SELU(inplace=True))])
        elif activation == 'sigmoid':
            net.update([(f'fc{i}_sigmoid', nn.Sigmoid())])
        elif activation == 'tanh':
            net.update([(f'fc{i}_tanh', nn.Tanh())])
        else:
            raise ValueError('activation not supported')

    net.update([('fc', nn.Linear(width[-1], nout, bias=True))])

    if outactivation == 'none':
        pass
    elif outactivation == 'tanh':
        net.update([('out', nn.Tanh())])
    elif outactivation == 'sigmoid':
        net.update([('out', nn.Sigmoid())])
    else:
        raise ValueError('out activation not supported')

    return nn.Sequential(net)


class MLP_G_Img(nn.Module):
    """
    Generator for images (with channel)

    activation: leakyrelu | relu | elu | selu | sigmoid | tanh
    normalize: batch | instance | none
    width: int | tuple | list (with len = depth - 1)
    outactivation: none | tanh | sigmoid
    """
    def __init__(self, nz, isize, nc, width=128, depth=3,
                 activation='leakyrelu', normalize='none', outactivation='tanh', ngpu=1):
        super().__init__()
        self.ngpu       = ngpu
        self.nz         = nz
        self.isize      = isize
        self.nc         = nc
        self.nx         = nc * isize * isize
        self.width      = width
        self.depth      = depth
        self.activation = activation
        self.normalize  = normalize
        self.main       = build_mlp(nz, self.nx, width, depth, activation, normalize, outactivation)

    def forward(self, input):
        input = input.view(input.size(0), input.size(1))
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(output.size(0), self.nc, self.isize, self.isize)


class MLP_D_Img(nn.Module):
    """
    Discriminator for images (with channel)

    activation: leakyrelu | relu | elu | selu | sigmoid | tanh
    normalize: batch | instance | none
    width: int | tuple | list (with len = depth - 1)
    outactivation: none | tanh | sigmoid
    """
    def __init__(self, isize, nc, outdim=1, width=128, depth=3,
                 activation='leakyrelu', normalize='none', outactivation='sigmoid', ngpu=1):
        super().__init__()
        self.ngpu       = ngpu
        self.isize      = isize
        self.nc         = nc
        self.nx         = nc * isize * isize
        self.outdim     = outdim
        self.width      = width
        self.depth      = depth
        self.activation = activation
        self.normalize  = normalize
        self.main       = build_mlp(self.nx, outdim, width, depth, activation, normalize, outactivation)

    def forward(self, input):
        input = input.view(input.size(0),
                           input.size(1) * input.size(2) * input.size(3))
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(output.size(0), self.outdim)


class MLP_G_Toy(nn.Module):
    """
    Toy generator (without channel)

    activation: leakyrelu | relu | elu | selu | sigmoid | tanh
    normalize: batch | instance | none
    width: int | tuple | list (with len = depth - 1)
    outactivation: none | tanh | sigmoid
    """
    def __init__(self, nz, nx, width=128, depth=3,
                 activation='leakyrelu', normalize='none', outactivation='none', ngpu=1):
        super().__init__()
        self.ngpu       = ngpu
        self.nz         = nz
        self.nx         = nx
        self.width      = width
        self.depth      = depth
        self.activation = activation
        self.normalize  = normalize
        self.main       = build_mlp(nz, nx, width, depth, activation, normalize, outactivation)

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(output.size(0), self.nx)


class MLP_D_Toy(nn.Module):
    """
    Toy discriminator (without channel)

    activation: leakyrelu | relu | elu | selu | sigmoid | tanh
    normalize: batch | instance | none
    width: int | tuple | list (with len = depth - 1)
    outactivation: none | tanh | sigmoid
    """
    def __init__(self, nx, outdim=1, width=128, depth=3,
                 activation='leakyrelu', normalize='none', outactivation='none', ngpu=1):
        super().__init__()
        self.ngpu       = ngpu
        self.nx         = nx
        self.outdim     = outdim
        self.width      = width
        self.depth      = depth
        self.activation = activation
        self.normalize  = normalize
        self.main       = build_mlp(nx, outdim, width, depth, activation, normalize, outactivation)

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(output.size(0), self.outdim)

# random initialization
def weights_init_msra(m):
    """
    Modified version of msra init
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, np.sqrt(2.0 / m.in_features))
        m.bias.data.fill_(0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)
    # elif classname.find('InstanceNorm') != -1:
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.fill_(0.0)
