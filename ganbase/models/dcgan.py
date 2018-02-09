"""
DCGAN
"""
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np

def get_activation(activation):
    if activation == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif activation == 'relu':
        return nn.ReLU(True)
    elif activation == 'elu':
        return nn.ELU(inplace=True)
    elif activation == 'selu':
        return nn.SELU(inplace=True)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError('activation not supported')


def get_normalize(normalize, nf):
    if normalize == 'batch':
        return nn.BatchNorm2d(nf)
    elif normalize == 'instance':
        return nn.InstanceNorm2d(nf)
    else:
        raise ValueError('normalize layer not supported')


class DCGAN_D(nn.Module):
    """
    activation: leakyrelu | relu | elu | selu | sigmoid | tanh
    normalize = 'none | batch | instance'
    outactivation: none | tanh | sigmoid | elu
    """
    def __init__(self, isize, nc, ndf, n_extra_layers=0, n_extra_conv=0, activation='leakyrelu',
                 normalize='none', outactivation='none', outdim=1, ngpu=1):
        super().__init__()
        self.isize          = isize
        self.nc             = nc
        self.ndf            = ndf
        self.n_extra_layers = n_extra_layers
        self.n_extra_conv   = n_extra_conv
        self.activation     = activation
        self.normalize      = normalize
        self.outactivation  = outactivation
        self.outdim         = outdim
        self.bias           = True if normalize == 'none' else False
        self.ngpu           = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module(f'initial.conv.{nc}-{ndf}',
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=self.bias))

        main.add_module(f'initial.{ndf}.{activation}',
                        get_activation(activation))

        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(f'extra{t}.{cndf}.conv',
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=self.bias))

            if normalize != 'none':
                main.add_module(f'extra{t}.{cndf}.{normalize}norm',
                                get_normalize(normalize, cndf))

            main.add_module(f'extra{t}.{cndf}.{activation}',
                            get_activation(activation))

        while csize > 4:
            in_feat  = cndf
            out_feat = cndf * 2
            main.add_module(f'pyramid.{in_feat}-{out_feat}.conv',
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=self.bias))

            if normalize != 'none':
                main.add_module(f'pyramid.{out_feat}.{normalize}norm',
                                get_normalize(normalize, out_feat))

            main.add_module(f'pyramid.{out_feat}.{activation}',
                            get_activation(activation))

            # extra conv
            for t in range(n_extra_conv):
                main.add_module(f'pyramid.{out_feat}.extraconv{t}.conv',
                                nn.Conv2d(out_feat, out_feat, 3, 1, 1, bias=self.bias))

                if normalize != 'none':
                    main.add_module(f'pyramid.{out_feat}.extraconv{t}.{normalize}norm',
                                    get_normalize(normalize, out_feat))

                main.add_module(f'pyramid.{out_feat}.extraconv{t}.{activation}',
                                get_activation(activation))

            cndf  = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module(f'final.{cndf}-{outdim}.conv',
                        nn.Conv2d(cndf, outdim, 4, 1, 0, bias=self.bias))

        if outactivation != 'none':
            main.add_module(f'final.{cndf}-{outdim}.{outactivation}',
                            get_activation(outactivation))
        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        output = output.view(output.size(0), self.outdim)

        return output


class DCGAN_G(nn.Module):
    """
    activation: leakyrelu | relu | elu | selu | sigmoid | tanh
    normalize = 'none | batch | instance'
    """
    def __init__(self, isize, nc, nz, ngf, n_extra_layers=0, n_extra_conv=0,
                 activation='leakyrelu', normalize='batch', ngpu=1):
        super().__init__()
        self.isize          = isize
        self.nc             = nc
        self.nz             = nz
        self.ngf            = ngf
        self.n_extra_layers = n_extra_layers
        self.n_extra_conv   = n_extra_conv
        self.activation     = activation
        self.normalize      = normalize
        self.bias           = True if normalize == 'none' else False
        self.ngpu           = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf   = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z (nc x nz x 1 x 1), going into a convolution
        main.add_module(f'initial.{nz}-{cngf}.convt',
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=self.bias))

        if normalize != 'none':
            main.add_module(f'initial.{cngf}.{normalize}norm',
                            get_normalize(normalize, cngf))

        main.add_module(f'initial.{cngf}.{activation}',
                        get_activation(activation))

        csize, cngf = 4, cngf
        while csize < isize//2:
            main.add_module(f'pyramid.{cngf}-{cngf//2}.convt',
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=self.bias))

            if normalize != 'none':
                main.add_module(f'pyramid.{cngf//2}.{normalize}norm',
                                get_normalize(normalize, cngf//2))

            main.add_module(f'pyramid.{cngf//2}.{activation}',
                            get_activation(activation))

            # extra conv
            for t in range(n_extra_conv):
                main.add_module(f'pyramid.{cngf//2}.extraconv{t}.conv',
                                nn.Conv2d(cngf//2, cngf//2, 3, 1, 1, bias=self.bias))

                if normalize != 'none':
                    main.add_module(f'pyramid.{cngf//2}.extraconv{t}.{normalize}norm',
                                    get_normalize(normalize, cngf//2))

                main.add_module(f'pyramid.{cngf//2}.extraconv{t}.{activation}',
                                get_activation(activation))

            cngf  = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(f'extra{t}.{cngf}.conv',
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=self.bias))

            if normalize != 'none':
                main.add_module(f'extra{t}.{cngf}.{normalize}norm',
                                get_normalize(normalize, cngf))

            main.add_module(f'extra{t}.{cngf}.{activation}',
                            get_activation(activation))

        main.add_module(f'final.{cngf}-{nc}.convt',
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=self.bias))
        main.add_module(f'final.{nc}.tanh'.format(),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if input.ndimension() == 2:
            input = input.view(input.size(0), input.size(1), 1, 1)
        assert input.size() == (input.size(0), self.nz, 1, 1)

        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


###############################################################################

def weights_init(m):
    """
    init
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
