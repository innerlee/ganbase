# -*- coding: utf-8 -*-

import torch.nn as nn
import math


class Generator(nn.Module):
    def __init__(self, z_size=100, init_square_size=8, ngf=128, img_size=64, img_channel=3,
                 norm_layer=None, act_layer=nn.ELU()):
        super().__init__()
        self.init_conv_size = init_square_size
        self.ngf = ngf
        self.n_repeat = int(math.log(img_size, 2)) - int(math.log(init_square_size, 2)) + 1
        self.fc = nn.Linear(z_size, init_square_size * init_square_size * ngf)

        main = []
        for i in range(self.n_repeat):
            main.append(nn.Conv2d(ngf, ngf, 3, 1, 1))
            if norm_layer is not None:
                main.append(norm_layer(ngf))
            main.append(act_layer)
            main.append(nn.Conv2d(ngf, ngf, 3, 1, 1))
            if norm_layer is not None:
                main.append(norm_layer(ngf))
            main.append(act_layer)
            if i < self.n_repeat - 1:
                main.append(nn.Upsample(scale_factor=2.0, mode='nearest'))
        main.append(nn.Conv2d(ngf, img_channel, 3, 1, 1))
        self.main = nn.Sequential(*main)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.ngf, self.init_conv_size, self.init_conv_size)
        x = self.main(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, z_size=100, init_square_size=8, ndf=128, img_size=64, img_channel=3,
                 norm_layer=None, act_layer=nn.ELU()):
        super().__init__()

        self.init_square_size = init_square_size
        self.n_repeat = int(math.log(img_size, 2)) - int(math.log(init_square_size, 2)) + 1
        self.ndf = ndf
        encoder = []
        encoder.append(nn.Conv2d(img_channel, ndf, 3, 1, 1))
        for i in range(self.n_repeat):
            c_ndf, t_ndf = (ndf * (i + 1), ndf * (i + 2)) if i < self.n_repeat - 1 else (ndf * (i + 1), ndf * (i + 1))
            encoder.append(nn.Conv2d(c_ndf, c_ndf, 3, 1, 1))
            if norm_layer is not None:
                encoder.append(norm_layer(c_ndf))
            encoder.append(act_layer)
            encoder.append(nn.Conv2d(c_ndf, t_ndf, 3, 1, 1))
            if norm_layer is not None:
                encoder.append(norm_layer(t_ndf))
            encoder.append(act_layer)
            if i < self.n_repeat - 1:
                encoder.append(nn.Conv2d(t_ndf, t_ndf, 3, 2, 1))
                if norm_layer is not None:
                    encoder.append(norm_layer(t_ndf))
                encoder.append(act_layer)
        self.encoder = nn.Sequential(*encoder)

        self.encoder_fc = nn.Linear(self.init_square_size * self.init_square_size * ndf * self.n_repeat, z_size)
        self.decoder_fc = nn.Linear(z_size, self.init_square_size * self.init_square_size * ndf)

        decoder = []
        for i in range(self.n_repeat):
            decoder.append(nn.Conv2d(ndf, ndf, 3, 1, 1))
            if norm_layer is not None:
                decoder.append(norm_layer(ndf))
            decoder.append(act_layer)
            decoder.append(nn.Conv2d(ndf, ndf, 3, 1, 1))
            if norm_layer is not None:
                decoder.append(norm_layer(ndf))
            decoder.append(act_layer)
            if i < self.n_repeat - 1:
                decoder.append(nn.Upsample(scale_factor=2.0, mode='nearest'))
        decoder.append(nn.Conv2d(ndf, img_channel, 3, 1, 1))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.n_repeat * self.init_square_size * self.init_square_size * self.ndf)
        x = self.encoder_fc(x)
        x = self.decoder_fc(x)
        x = x.view(-1, self.ndf, self.init_square_size, self.init_square_size)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    netG = Generator(z_size=100, img_size=128, init_square_size=8, img_channel=3)
    import torch
    x = torch.rand(10, 100)
    y = netG(x)
    print(y.size())
    netD = Discriminator(z_size=100, img_size=128, init_square_size=8, img_channel=3)
    rec_y = netD(y)
    print(rec_y.size())
