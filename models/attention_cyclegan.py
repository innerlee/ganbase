# -*- coding: utf-8 -*-

import torch.nn as nn
import functools
import math


def conv_norm_relu(in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                   norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True), padding_type='reflect',
                   with_norm=True, with_act=True):
    conv_block = []
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func != nn.BatchNorm2d
    else:
        use_bias = norm_layer != nn.BatchNorm2d
    padding = int(math.ceil((kernel_size * dilation - dilation + 1 - stride) / 2.))
    if padding_type == 'zero':
        p = padding
    else:
        p = 0
    if padding_type == 'reflect':
        conv_block += [nn.ReflectionPad2d(padding)]
    elif padding_type == 'replicate':
        conv_block += [nn.ReplicationPad2d(padding)]
    conv_block += [nn.Conv2d(in_channels, out_channels, kernel_size, stride, p, dilation, bias=use_bias)]
    if with_norm:
        conv_block += [norm_layer(out_channels)]
    if with_act:
        conv_block += [act_layer]
    return nn.Sequential(*conv_block)


def conv_t_norm_relu(in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                     norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True), with_norm=True, with_act=True):
    conv_block = []
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func != nn.BatchNorm2d
    else:
        use_bias = norm_layer != nn.BatchNorm2d
    assert kernel_size >= stride, 'kernel should be larger than kernel_size'
    size_diff = kernel_size - dilation + 1 - stride
    padding = int(math.ceil(size_diff / 2.0))
    output_padding = size_diff % 2

    conv_block += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding,
                                      dilation=dilation, output_padding=output_padding, bias=use_bias)]
    if with_norm:
        conv_block += [norm_layer(out_channels)]
    if with_act:
        conv_block += [act_layer]
    return nn.Sequential(*conv_block)


class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, act_layer, padding_type, use_dropout, with_norm=True):
        super(ResnetBlock, self).__init__()
        conv_block = [conv_norm_relu(dim, dim, 3, 1, 1, norm_layer, act_layer, padding_type, with_norm=with_norm)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [conv_norm_relu(dim, dim, 3, 1, 1, norm_layer, act_layer, padding_type, with_norm=with_norm)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True),
                 use_dropout=False, n_blocks=6, padding_type='reflect', with_norm=True):
        super(ResnetGenerator, self).__init__()

        assert (n_blocks >= 0)

        model = [conv_norm_relu(input_nc, ngf, 7, 1, 1, norm_layer, act_layer, padding_type, with_norm=with_norm)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [conv_norm_relu(ngf * mult, ngf * mult * 2, 3, 2, 1, norm_layer, act_layer, padding_type,
                                     with_norm=with_norm)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            model += [ResnetBlock(ngf * mult, norm_layer, act_layer, padding_type, use_dropout, with_norm=with_norm)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += conv_t_norm_relu(ngf * mult, int(ngf * mult / 2), 3, 2, 1, norm_layer, act_layer,
                                      with_norm=with_norm)

        model += [conv_norm_relu(ngf, output_nc, 3, 1, 1, norm_layer, act_layer, padding_type, False, False)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if norm_layer is not None:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        if norm_layer is not None:
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)]
        else:
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                nn.LeakyReLU(0.2, True)]
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class AttentionNet(nn.Module):
    def __init__(self, input_nc, output_nc, naf=64, norm_layer=nn.BatchNorm2d, padding_type='reflect',
                 act_layer=nn.ReLU(True), use_dropout=False):
        super(AttentionNet, self).__init__()

        model = [conv_norm_relu(input_nc, naf, 7, 2, 1, norm_layer, act_layer, padding_type)]
        model += [conv_norm_relu(naf, naf * 2, 3, 2, 1, norm_layer, act_layer, padding_type)]
        model += [ResnetBlock(naf * 2, norm_layer, act_layer, padding_type, use_dropout)]
        model += [nn.Upsample(mode='nearest', scale_factor=2.0)]
        model += [conv_norm_relu(naf * 2, naf * 2, 3, 1, 1, norm_layer, act_layer, padding_type)]
        model += [nn.Upsample(mode='nearest', scale_factor=2.0)]
        model += [conv_norm_relu(naf * 2, naf, 3, 1, 1, norm_layer, act_layer, padding_type)]
        model += [conv_norm_relu(naf, output_nc, 7, 1, 1, norm_layer, act_layer, padding_type, False, False)]
        model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


if __name__ == '__main__':
    netG = ResnetGenerator(3, 3, 32, nn.BatchNorm2d, nn.ReLU(True), False, 9)
    netD = NLayerDiscriminator(3, 64, 3, nn.BatchNorm2d)
    netA = AttentionNet(3, 1, 32, nn.BatchNorm2d)
    print(netG)
    print(netD)
    print(netA)

    import torch

    x = torch.randn(4, 3, 256, 256)
    y = netG(x)
    z = netD(y)
    a = netA(x)

    print(x.size())
    print(y.size())
    print(z.size())
    print(a.size())
    print(x.max(), x.min())
    print(y.max(), y.min())
    print(z.max(), z.min())
    print(a.max(), a.min())
