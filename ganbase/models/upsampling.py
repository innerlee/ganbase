"""
Upsampling + Conv + Conv
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
        raise ValueError(f'activation `{activation}` not supported')


def get_pooling(pooling):
    if pooling == 'max':
        return nn.MaxPool2d(2, 2)
    elif pooling == 'avg':
        return nn.AvgPool2d(2, 2)
    else:
        raise ValueError(f'pooling `{pooling}` not supported')


def get_normalize(normalize, nf):
    if normalize == 'batch':
        return nn.BatchNorm2d(nf)
    elif normalize == 'instance':
        return nn.InstanceNorm2d(nf)
    else:
        raise ValueError('normalize layer not supported')


class DOWNSAMPLE_D(nn.Module):
    """
    downsample: mean | max
    activation: leakyrelu | relu | elu | selu | sigmoid | tanh
    normalize = 'none | batch | instance'
    outactivation: none | tanh | sigmoid | elu
    """

    def __init__(self,
                 imsize,
                 imchannel,
                 netwidth,
                 extraconv=1,
                 pooling='avg',
                 activation='leakyrelu',
                 normalize='none',
                 outactivation='none',
                 outdim=1,
                 ngpu=1):
        super().__init__()
        #region yapf: disable
        self.imsize         = imsize
        self.imchannel      = imchannel
        self.netwidth       = netwidth
        self.extraconv      = extraconv
        self.activation     = activation
        self.normalize      = normalize
        self.outactivation  = outactivation
        self.outdim         = outdim
        self.bias           = True if normalize == 'none' else False
        self.ngpu           = ngpu
        #region yapf: enable
        assert imsize % 4 == 0, "imsize has to be a multiple of 4"

        main = nn.Sequential()

        insize, inchannel, outchannel = imsize, imchannel, netwidth

        while insize > 4:
            main.add_module(f'pyramid.{inchannel}-{outchannel}.conv',
                            nn.Conv2d(
                                inchannel, outchannel, 3, 1, 1,
                                bias=self.bias))

            if normalize != 'none':
                main.add_module(
                    f'pyramid.{inchannel}-{outchannel}.{normalize}norm',
                    get_normalize(normalize, outchannel))

            main.add_module(f'pyramid.{inchannel}-{outchannel}.{activation}',
                            get_activation(activation))

            # extra convs
            for t in range(extraconv):
                main.add_module(f'pyramid.{outchannel}.conv{t}.conv',
                                nn.Conv2d(
                                    outchannel,
                                    outchannel,
                                    3,
                                    1,
                                    1,
                                    bias=self.bias))

                if normalize != 'none':
                    main.add_module(
                        f'pyramid.{outchannel}.conv{t}.{normalize}norm',
                        get_normalize(normalize, outchannel))

                main.add_module(f'pyramid.{outchannel}.conv{t}.{activation}',
                                get_activation(activation))

            main.add_module(f'pyramid.{outchannel}.pool', get_pooling(pooling))

            insize //= 2
            inchannel = outchannel
            outchannel *= 2

        assert insize == 4

        # state size. bs x outchannel x 4 x 4
        main.add_module(f'final.{inchannel}-{outdim}.conv',
                        nn.Conv2d(inchannel, outdim, 4, 1, 0, bias=True))

        if outactivation != 'none':
            main.add_module(f'final.{outchannel}-{outdim}.{outactivation}',
                            get_activation(outactivation))
        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        output = output.view(output.size(0), self.outdim)

        return output


class UPSAMPLE_G(nn.Module):
    """
    upsample: nearest | bilinear
    activation: leakyrelu | relu | elu | selu | sigmoid | tanh
    normalize = 'none | batch | instance'
    """

    def __init__(self,
                 imsize,
                 imchannel,
                 nz,
                 width,
                 extraconv=0,
                 upsampling='nearest',
                 activation='leakyrelu',
                 normalize='batch',
                 ngpu=1):
        super().__init__()
        #region yapf: disable
        self.imsize         = imsize
        self.imchannel      = imchannel
        self.nz             = nz
        self.width          = width
        self.extraconv      = extraconv
        self.activation     = activation
        self.normalize      = normalize
        self.bias           = True if normalize == 'none' else False
        self.ngpu           = ngpu
        #region yapf: enable
        assert imsize % 4 == 0, "imsize has to be a multiple of 4"

        inchannel, insize = width, 8
        while insize < imsize:
            inchannel = inchannel * 2
            insize = insize * 2

        main = nn.Sequential()
        # input is Z (bs x nz x 1 x 1), going into a convolution
        main.add_module(f'initial.{nz}-{inchannel}.convt',
                        nn.ConvTranspose2d(
                            nz, inchannel, 4, 1, 0, bias=self.bias))

        if normalize != 'none':
            main.add_module(f'initial.{inchannel}.{normalize}norm',
                            get_normalize(normalize, inchannel))

        main.add_module(f'initial.{inchannel}.{activation}',
                        get_activation(activation))

        insize, outchannel = 4, inchannel // 2
        while insize < imsize:
            main.add_module(f'pyramid.{inchannel}.upsample',
                            nn.Upsample(scale_factor=2, mode=upsampling))

            # extra conv
            for t in range(extraconv):
                main.add_module(f'pyramid.{inchannel}.conv{t}.conv',
                                nn.Conv2d(
                                    inchannel,
                                    inchannel,
                                    3,
                                    1,
                                    1,
                                    bias=self.bias))

                if normalize != 'none':
                    main.add_module(
                        f'pyramid.{inchannel}.conv{t}.{normalize}norm',
                        get_normalize(normalize, inchannel))

                main.add_module(f'pyramid.{inchannel}.conv{t}.{activation}',
                                get_activation(activation))

            main.add_module(f'pyramid.{inchannel}-{outchannel}.conv',
                            nn.Conv2d(
                                inchannel, outchannel, 3, 1, 1,
                                bias=self.bias))
            if normalize != 'none':
                main.add_module(
                    f'pyramid.{inchannel}-{outchannel}.{normalize}norm',
                    get_normalize(normalize, outchannel))

            main.add_module(f'pyramid.{inchannel}-{outchannel}.{activation}',
                            get_activation(activation))

            insize = insize * 2
            inchannel //= 2
            outchannel = inchannel // 2

        assert insize == imsize

        main.add_module(f'pyramid.{inchannel}-{imchannel}.conv',
                        nn.Conv2d(inchannel, imchannel, 3, 1, 1, bias=True))

        main.add_module(f'final.{imchannel}.tanh'.format(), nn.Tanh())
        self.main = main

    def forward(self, input):
        if input.ndimension() == 2:
            input = input.view(input.size(0), input.size(1), 1, 1)
        assert input.size() == (input.size(0), self.nz, 1, 1)

        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)
        return output
