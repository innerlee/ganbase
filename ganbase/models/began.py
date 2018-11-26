"""
BEGAN
"""

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


class BEGAN_D(nn.Module):
    """
    activation: leakyrelu | relu | elu | selu | sigmoid | tanh
    normalize = 'none | batch | instance'

    """

    def __init__(self,
                 imsize,
                 imchannel,
                 nz,
                 width=128,
                 activation='elu',
                 normalize='none'):
        super().__init__()
        #region yapf: disable
        self.imsize         = imsize
        self.imchannel      = imchannel
        self.nz             = nz
        self.width          = width
        self.activation     = activation
        self.normalize      = normalize
        self.bias           = True if normalize == 'none' else False
        #region yapf: enable
        assert imsize % 16 == 0, "imsize has to be a multiple of 16"

        encoder = []
        repeat_num = int(np.log2(imsize)) - 2
        # Encoder
        encoder.append(nn.Conv2d(imchannel, width, 3, 1, 1))
        encoder.append(get_activation(activation))

        prev_channel_num = width
        for idx in range(repeat_num):
            channel_num = width * (idx + 1)
            encoder.append(nn.Conv2d(prev_channel_num, prev_channel_num, 3, 1, 1))
            if normalize != 'none':
                encoder.append(get_normalize(normalize, channel_num))
            encoder.append(get_activation(activation))

            encoder.append(nn.Conv2d(prev_channel_num, channel_num, 3, 1, 1))
            if normalize != 'none':
                encoder.append(get_normalize(normalize, channel_num))
            encoder.append(get_activation(activation))


            if idx < repeat_num - 1:
                encoder.append(nn.Conv2d(channel_num, channel_num, 1, 1, 0))
                encoder.append(nn.AvgPool2d(2, 2))
            else:
                encoder.append(nn.Conv2d(channel_num, channel_num, 3, 1, 1))

            if normalize != 'none':
                encoder.append(get_normalize(normalize, channel_num))

            prev_channel_num = channel_num

        self.encoder=torch.nn.Sequential(*encoder)

        self.conv1_output_dim = [channel_num, 8, 8]

        self.fc1 = nn.Linear(8 * 8 * channel_num, nz)

        # Decoder
        self.conv2_input_dim = [width, 8, 8]
        self.fc2 = nn.Linear(nz, 8*8*width)

        decoder = []

        for idx in range(repeat_num):
            decoder.append(nn.Conv2d(self.width, self.width, 3, 1, 1))
            if normalize != 'none':
                decoder.append(get_normalize(normalize, self.width))
            decoder.append(get_activation(activation))

            decoder.append(nn.Conv2d(self.width, self.width, 3, 1, 1))
            if normalize != 'none':
                decoder.append(get_normalize(normalize, self.width))
            decoder.append(get_activation(activation))

            if idx < repeat_num - 1:
                decoder.append(nn.UpsamplingNearest2d(scale_factor=2))

        decoder.append(nn.Conv2d(self.width, imchannel, 3, 1, 1))
        decoder.append(get_activation('tanh'))
        self.decoder = torch.nn.Sequential(*decoder)



    def forward(self, input):
        conv1_out = self.encoder(input).view(-1, np.prod(self.conv1_output_dim))
        fc1_out = self.fc1(conv1_out)

        fc2_out = self.fc2(fc1_out).view([-1] + self.conv2_input_dim)
        conv2_out = self.decoder(fc2_out)

        return conv2_out


class BEGAN_G(nn.Module):
    """
    activation: leakyrelu | relu | elu | selu | sigmoid | tanh
    normalize = 'none | batch | instance'
    """

    def __init__(self,
                 imsize,
                 imchannel,
                 nz,
                 width=128,
                 activation='elu',
                 normalize='none'):
        super().__init__()
        #region yapf: disable
        self.imsize           = imsize
        self.imchannel        = imchannel
        self.nz               = nz
        self.width            = width
        self.activation       = activation
        self.normalize        = normalize
        self.bias             = True if normalize == 'none' else False
        #region yapf: enable
        assert imsize % 16 == 0, "imsize has to be a multiple of 16"

        self.fc = nn.Linear(self.nz, 8 * 8 * width)

        layers = []
        repeat_num=int(np.log2(imsize))-2
        for idx in range(repeat_num):
            layers.append(nn.Conv2d(self.width, self.width, 3, 1, 1))
            if normalize != 'none':
                layers.append(get_normalize(normalize, self.width))
            layers.append(get_activation(activation))

            layers.append(nn.Conv2d(self.width, self.width, 3, 1, 1))
            if normalize != 'none':
                layers.append(get_normalize(normalize, self.width))
            layers.append(get_activation(activation))

            if idx < repeat_num - 1:
                layers.append(nn.UpsamplingNearest2d(scale_factor=2))

        layers.append(nn.Conv2d(self.width, imchannel, 3, 1, 1))
        layers.append(get_activation('tanh'))
        self.conv = torch.nn.Sequential(*layers)


    def forward(self, input):
        assert input.size() == (input.size(0), self.nz)
        fc_out = self.fc(input).view([-1,self.width,8,8])
        output = self.conv(fc_out)
        return output



