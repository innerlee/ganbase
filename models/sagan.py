# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class Generator(nn.Module):
    """Generator."""

    def __init__(self, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer = []
        layer1 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num  # 8
        layer.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer.append(nn.BatchNorm2d(conv_dim * mult))
        layer.append(nn.ReLU())

        curr_dim = conv_dim * mult

        for i in range(repeat_num - 1):
            layer.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer.append(nn.ReLU())

            curr_dim = int(curr_dim / 2)

        layer1.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer1.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer1.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        self.l = nn.Sequential(*layer)
        self.l1 = nn.Sequential(*layer1)

        # last.append(nn.ConvTranspose2d(curr_dim//2, 3, 4, 2, 1))
        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(128, 'relu')
        self.attn2 = Self_Attn(64, 'relu')

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.l(z)
        out, p1 = self.attn1(out)
        out = self.l1(out)
        out, p2 = self.attn2(out)
        out = self.last(out)

        return out, p1, p2


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer = []
        layer1 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3

        layer.append(SpectralNorm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
        layer.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        for i in range(repeat_num - 1):
            layer.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer.append(nn.LeakyReLU(0.1))
            curr_dim = curr_dim * 2

        layer1.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        self.l = nn.Sequential(*layer)
        self.l1 = nn.Sequential(*layer1)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(curr_dim // 2, 'relu')
        self.attn2 = Self_Attn(curr_dim, 'relu')

    def forward(self, x):
        out = self.l(x)
        out, p1 = self.attn1(out)
        out = self.l1(out)
        out, p2 = self.attn2(out)
        out = self.last(out)

        return out.squeeze(3).squeeze(2), p1, p2


if __name__ == '__main__':
    netG = Generator(image_size=128, z_dim=100, conv_dim=64)
    netD = Discriminator(image_size=128, conv_dim=64)

    x = torch.rand(10, 100)
    y = netG(x)
    print(y[0].size())
    rec_y = netD(y[0])
    print(rec_y[0].size())
