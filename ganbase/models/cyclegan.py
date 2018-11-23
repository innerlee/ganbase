"""
CycleGAN
"""
import torch.nn as nn


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


# PatchGAN discriminator Structure
class CycleGAN_D(nn.Module):
    """
    activation: leakyrelu | relu | elu | selu | sigmoid | tanh
    normalize = 'none | batch | instance'
    outactivation: none | tanh | sigmoid | elu
    """

    def __init__(self,
                 imsize,
                 imchannel,
                 netwidth,
                 extralayers,
                 activation='leakyrelu',
                 normalize='none',
                 outactivation='none',
                 outdim=1,
                 ngpu=1):
        super().__init__()
        self.imsize = imsize
        self.imchannel = imchannel
        self.netwidth = netwidth
        self.extralayers = extralayers
        self.activation = activation
        self.normalize = normalize
        self.outactivation = outactivation
        self.outdim = outdim
        self.bias = True if normalize in ('none', 'instance') else False
        self.ngpu = ngpu
        # region yapf: enable
        assert imsize % 4 == 0, "imsize has to be a multiple of 4"
        main = nn.Sequential()

        # input is bs x imchannel x imsize x imsize
        main.add_module(f'initial_conv_{imchannel}-{netwidth}',
                        nn.Conv2d(
                            imchannel, netwidth, 4, 2, 1, bias=self.bias))

        main.add_module(f'initial_{netwidth}_{activation}',
                        get_activation(activation))

        n_latter = 1
        n_prev = 1
        # 这部分是下采样，像素 1/4

        for t in range(1, self.extralayers):
            n_prev = n_latter
            n_latter = min(2 ** t, 8)  # channel不能太多， 如果太多就保持channel数不变
            main.add_module(f'pyramid_{netwidth*n_prev}-{self.netwidth*n_latter}-{t}_conv',
                            nn.Conv2d(
                                netwidth * n_prev,
                                self.netwidth * n_latter,
                                4,
                                2,
                                1,
                                bias=self.bias))
            if normalize != 'none':
                main.add_module(
                    f'pyramid_{netwidth*n_latter}-{self.netwidth*n_latter}-{t}_{normalize}norm',
                    get_normalize(normalize, netwidth * n_latter))

            main.add_module(
                f'pyramid_{netwidth*n_latter}-{self.netwidth*n_latter}-{t}_{activation}',
                get_activation(activation))
        n_prev = n_latter
        n_latter = min(2 ** extralayers, 8)

        main.add_module(f'last_not_least_{netwidth*n_prev}-{self.netwidth*n_latter}_conv',
                        nn.Conv2d(
                            netwidth * n_prev,
                            self.netwidth * n_latter,
                            4,
                            2,
                            1,
                            bias=self.bias))
        if normalize != 'none':
            main.add_module(
                f'last_not_least_{netwidth*n_prev}-{self.netwidth*n_latter}_{normalize}norm',
                get_normalize(normalize, netwidth * n_latter))
        main.add_module(
            f'last_not_least_{netwidth*n_prev}-{self.netwidth*n_latter}_{activation}',
            get_activation(activation))

        main.add_module(f'final_{netwidth*n_latter}-{outdim}_conv',
                        nn.Conv2d(netwidth * n_latter, outdim, 4, 1, 0, bias=self.bias))

        if outactivation != 'none':
            main.add_module(f'final_{cndf}-{outdim}_{outactivation}',
                            get_activation(outactivation))
        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input,
                                               range(self.ngpu))
        else:
            output = self.main(input)

        return output


# Down-resnet-Up Structure
class CycleGAN_G(nn.Module):
    def __init__(self,
                 imsize,
                 imchannel,
                 width,
                 n_downsampling=2,
                 extralayers=6,
                 activation='leakyrelu',
                 normalize='batch',
                 ngpu=1,
                 use_dropout=False):
        super().__init__()
        # region yapf: disable
        self.imsize = imsize
        self.imchannel = imchannel
        self.width = width
        self.n_downsampling = n_downsampling
        self.extralayers = extralayers
        self.activation = activation
        self.normalize = normalize
        self.bias = True if normalize in ('none', 'instance') else False
        self.ngpu = ngpu
        # region yapf: enable
        assert imsize % 4 == 0, "imsize has to be a multiple of 4"

        model = []

        model += [nn.Conv2d(self.imchannel, self.width, kernel_size=7, padding=3, bias=self.bias),
                  get_normalize(self.normalize, self.width),
                  get_activation(self.activation)]

        for i in range(self.n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(self.width * mult, self.width * mult * 2, kernel_size=3, stride=2, padding=1, bias=self.bias),
                get_normalize(self.normalize, width * mult * 2),
                get_activation(self.activation)]
        mult = 2 ** self.n_downsampling

        for i in range(self.extralayers):
            model += [ResnetBlock(width * mult, activation=self.activation, normalize=self.normalize,
                                  use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(width * mult, int(width * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1, bias=self.bias),
                      get_normalize(self.normalize, int(width * mult / 2)),
                      get_activation(self.activation)]

        model += [nn.Conv2d(width, imchannel, kernel_size=7, padding=3, bias=self.bias)]
        model += [nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResnetBlock(nn.Module):
    def __init__(self,
                 dim=64,
                 pad_type='reflect',
                 activation='leakyrelu',
                 normalize='batch',
                 use_dropout=False):
        super(ResnetBlock, self).__init__()
        acti_layer = get_activation(activation)
        norm_layer = get_normalize(normalize, dim)
        use_bias = True if normalize in ('none', 'instance') else False
        self.conv_block = self.make_block(dim, pad_type, acti_layer, norm_layer, use_dropout, use_bias)

    def make_block(self, dim, pad_type, acti_layer, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if pad_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif pad_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif pad_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding {:s} is not implemented'.format(pad_type))

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer,
                       acti_layer]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if pad_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif pad_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif pad_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding {:s} is not implemented'.format(pad_type))

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
