# -*- coding: utf-8 -*-

import torch, torch.nn as nn
import torch.nn.functional as F


class Conv_ReLU(nn.Module):

    def __init__(self, nin, nout, ks, ss, ps, has_bn, bn, relu, has_bias=False):
        super(Conv_ReLU, self).__init__()
        if has_bn:
            self.subnet = nn.Sequential(nn.Conv2d(nin, nout, kernel_size=ks, stride=ss, padding=ps, bias=has_bias),
                                        bn(nout), relu)
        else:
            self.subnet = nn.Sequential(nn.Conv2d(nin, nout, kernel_size=ks, stride=ss, padding=ps, bias=has_bias),
                                        relu)

    def forward(self, x):
        return self.subnet(x)


class DeConv_ReLU(nn.Module):

    def __init__(self, nin, nout, ks, ss, ps, has_bn, bn, relu, has_bias=False):
        super(DeConv_ReLU, self).__init__()
        if has_bn:
            self.subnet = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, kernel_size=ks, stride=ss, padding=ps, bias=has_bias), bn(nout), relu)
        else:
            self.subnet = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, kernel_size=ks, stride=ss, padding=ps, bias=has_bias), relu)

    def forward(self, x):
        return self.subnet(x)


class InferenceBatchSoftmax(nn.Module):

    def forward(self, input_):
        if not self.training:
            return F.softmax(input_, dim=-1)
        else:
            return input_


class Encoder(nn.Module):

    def __init__(self, input_nc, output_dim, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU(0.2, True)):
        super(Encoder, self).__init__()
        ks = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        ps = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
        ss = [(2, 2), (2, 2), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1)]
        nm = [64, 128, 256, 256, 512, 512, output_dim]
        self.s1 = nn.Sequential(Conv_ReLU(input_nc, nm[0], ks[0], ss[0], ps[0], False, norm_layer, act_layer, True))
        self.s2 = nn.Sequential(Conv_ReLU(nm[0], nm[1], ks[1], ss[1], ps[1], False, norm_layer, act_layer, True))
        self.s3 = nn.Sequential(Conv_ReLU(nm[1], nm[2], ks[2], ss[2], ps[2], True, norm_layer, act_layer, False),
                                Conv_ReLU(nm[2], nm[3], ks[3], ss[3], ps[3], False, norm_layer, act_layer, True))
        self.s4 = nn.Sequential(Conv_ReLU(nm[3], nm[4], ks[4], ss[4], ps[4], True, norm_layer, act_layer, False),
                                Conv_ReLU(nm[4], nm[5], ks[5], ss[5], ps[5], False, norm_layer, act_layer, True))
        self.s5 = nn.Sequential(Conv_ReLU(nm[5], nm[6], ks[6], ss[6], ps[6], True, norm_layer, act_layer, False))

    def forward(self, x):
        s1 = self.s1(x)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)
        s5 = self.s5(s4)
        return s5


class Decoder(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True)):
        super(Decoder, self).__init__()
        ks = [(4, 4), (4, 4), (3, 3), (4, 4), (3, 3), (4, 4), (4, 4)]
        ps = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
        ss = [(2, 2), (2, 2), (1, 1), (2, 2), (1, 1), (2, 2), (2, 2)]
        nm = [512, 512, 256, 256, 128, 64, output_nc]
        self.s1 = nn.Sequential(DeConv_ReLU(input_nc, nm[0], ks[0], ss[0], ps[0], True, norm_layer, act_layer, False))
        self.s2 = nn.Sequential(DeConv_ReLU(nm[0], nm[1], ks[1], ss[1], ps[1], False, norm_layer, act_layer, True),
                                DeConv_ReLU(nm[1], nm[2], ks[2], ss[2], ps[2], True, norm_layer, act_layer, False))
        self.s3 = nn.Sequential(DeConv_ReLU(nm[2], nm[3], ks[3], ss[3], ps[3], False, norm_layer, act_layer, True),
                                DeConv_ReLU(nm[3], nm[4], ks[4], ss[4], ps[4], True, norm_layer, act_layer, False))
        self.s4 = nn.Sequential(DeConv_ReLU(nm[4], nm[5], ks[5], ss[5], ps[5], False, norm_layer, act_layer, True))
        self.s5 = nn.Sequential(
            nn.ConvTranspose2d(nm[5], nm[6], kernel_size=ks[6], stride=ss[6], padding=ps[6], bias=True))

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=4.0, mode='nearest')
        s1 = self.s1(x)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)
        s5 = self.s5(s4)
        return s5.tanh()


class Rec_Tail(nn.Module):

    def __init__(self, nout, nClasses):
        super(Rec_Tail, self).__init__()
        self.nClasses = nClasses
        self.fc = nn.Conv2d(nout, nClasses, 1, 1)
        self.softmax = InferenceBatchSoftmax()

    def forward(self, x):
        res = self.softmax(self.fc(x))
        return res


class Vae_Tail(nn.Module):

    def __init__(self, nin, nout, norm_layer):
        super(Vae_Tail, self).__init__()
        self.nout = nout
        self.fc1 = nn.Sequential(nn.Conv2d(nin, nout * 2, kernel_size=1, stride=1, padding=0, bias=True))
        self.fc2 = nn.Sequential(nn.Conv2d(nout, nin, kernel_size=1, stride=1, padding=0, bias=True), norm_layer(nin),
                                 nn.ReLU(True))

    def encoder(self, x):
        res = self.fc1(x)
        mu = res[:, :self.nout, :, :]
        logvar = res[:, self.nout:, :, :]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def decoder(self, x):
        return self.fc2(x)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        c_code = self.reparametrize(mu, logvar)
        res = self.decoder(c_code)
        return mu, logvar, res


class NetI(nn.Module):

    def __init__(self, input_nc, output_dim, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU(0.2, True), nClasses=2):
        super(NetI, self).__init__()
        self.encoder = Encoder(input_nc, output_dim, norm_layer, act_layer)
        self.rec_tail = Rec_Tail(output_dim, nClasses)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.avg_pool(feat)
        res = self.rec_tail(feat)
        return feat, res


class NetA(nn.Module):

    def __init__(self, input_nc, output_dim, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU(0.2, True), vae_dim=100):
        super(NetA, self).__init__()
        self.encoder = Encoder(input_nc, output_dim, norm_layer, act_layer)
        self.vae_tail = Vae_Tail(output_dim, vae_dim, norm_layer)
        self.ave_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat = self.encoder(x)  ## N*512*1*W -> N*512*1*1
        # n, c, h, w = feat.size()
        global_feat = self.ave_pool(feat)
        mu, logvar, res = self.vae_tail(global_feat)
        # res = res.expand(-1, -1, h, w)
        return mu, logvar, res


class NetG(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU(True)):
        super(NetG, self).__init__()
        self.project = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False),
                                     norm_layer(512),
                                     nn.ReLU(True))
        self.decoder = Decoder(input_nc, output_nc, norm_layer, act_layer)

    def forward(self, x1, x2):
        x1_ = self.project(x1)
        return self.decoder(torch.cat((x1_, x2), 1))


class NetD(nn.Module):

    def __init__(self, input_nc, output_dim, norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU(0.2, True)):
        super(NetD, self).__init__()
        self.encoder = Encoder(input_nc, output_dim, norm_layer, act_layer)
        self.dis_tail = nn.Sequential(nn.Conv2d(output_dim, 1, kernel_size=1, stride=1, padding=0, bias=True),
                                      nn.Sigmoid())

    def forward(self, x):
        feat = self.encoder(x)
        res = self.dis_tail(feat)
        return feat, res


if __name__ == '__main__':
    netI = NetI(3, 512)
    netA = NetA(3, 512)

    input_img = torch.FloatTensor(4, 3, 128, 128)
    feat_I, res = netI(input_img)
    print('output of I', feat_I.size(), res.size())

    mu, logvar, feat_A = netA(input_img)
    print('output of A', mu.size(), logvar.size(), feat_A.size())

    netG = NetG(1024, 3)
    gen = netG(feat_I, feat_A)
    print('output pf G', gen.size())

# fuse_feat = torch.cat((feat_I, feat_A), 1)
# output_img = netG(fuse_feat)
# print(output_img.size())

# feat_D, res = netD(output_img)
# print(feat_D.size(), res.size())

# criterion_kl = KL_Loss()
# criterion_gan = GANLoss()

# kl_loss = criterion_kl(mu, logvar)
# gan_loss = criterion_gan(res, True)

# print(kl_loss, gan_loss)
