import torch, torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        # def lambda_rule(epoch):
        #     lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
        #     return lr_l
        def lambda_rule(cur_iter):
            lr = ((1. - float(cur_iter) / opt.niter) ** 0.9)
            return lr

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        if not torch.cuda.is_available():
            raise AssertionError
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


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

    def __init__(self, input_nc, output_dim, norm_layer='batch', relu_layer='leaky'):
        super(Encoder, self).__init__()
        NormLayer = get_norm_layer(norm_layer)
        if relu_layer == 'relu':
            ReLULayer = nn.ReLU(True)
        else:
            ReLULayer = nn.LeakyReLU(0.2, True)
        ks = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        ps = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
        ss = [(2, 2), (2, 2), (1, 1), (2, 2), (1, 1), (2, 2), (1, 1)]
        nm = [64, 128, 256, 256, 512, 512, output_dim]
        self.s1 = nn.Sequential(Conv_ReLU(input_nc, nm[0], ks[0], ss[0], ps[0], False, NormLayer, ReLULayer, True))
        self.s2 = nn.Sequential(Conv_ReLU(nm[0], nm[1], ks[1], ss[1], ps[1], False, NormLayer, ReLULayer, True))
        self.s3 = nn.Sequential(Conv_ReLU(nm[1], nm[2], ks[2], ss[2], ps[2], True, NormLayer, ReLULayer, False),
                                Conv_ReLU(nm[2], nm[3], ks[3], ss[3], ps[3], False, NormLayer, ReLULayer, True))
        self.s4 = nn.Sequential(Conv_ReLU(nm[3], nm[4], ks[4], ss[4], ps[4], True, NormLayer, ReLULayer, False),
                                Conv_ReLU(nm[4], nm[5], ks[5], ss[5], ps[5], False, NormLayer, ReLULayer, True))
        self.s5 = nn.Sequential(Conv_ReLU(nm[5], nm[6], ks[6], ss[6], ps[6], True, NormLayer, ReLULayer, False))

    def forward(self, x):
        s1 = self.s1(x)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)
        s5 = self.s5(s4)
        return s5


class Decoder(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer='batch', relu_layer='relu'):
        super(Decoder, self).__init__()
        NormLayer = get_norm_layer(norm_layer)
        if relu_layer == 'relu':
            ReLULayer = nn.ReLU(True)
        else:
            ReLULayer = nn.LeakyReLU(0.2, True)
        ks = [(4, 4), (4, 4), (3, 3), (4, 4), (3, 3), (4, 4), (4, 4)]
        ps = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
        ss = [(2, 2), (2, 2), (1, 1), (2, 2), (1, 1), (2, 2), (2, 2)]
        nm = [512, 512, 256, 256, 128, 64, output_nc]
        self.s1 = nn.Sequential(DeConv_ReLU(input_nc, nm[0], ks[0], ss[0], ps[0], True, NormLayer, ReLULayer, False))
        self.s2 = nn.Sequential(DeConv_ReLU(nm[0], nm[1], ks[1], ss[1], ps[1], False, NormLayer, ReLULayer, True),
                                DeConv_ReLU(nm[1], nm[2], ks[2], ss[2], ps[2], True, NormLayer, ReLULayer, False))
        self.s3 = nn.Sequential(DeConv_ReLU(nm[2], nm[3], ks[3], ss[3], ps[3], False, NormLayer, ReLULayer, True),
                                DeConv_ReLU(nm[3], nm[4], ks[4], ss[4], ps[4], True, NormLayer, ReLULayer, False))
        self.s4 = nn.Sequential(DeConv_ReLU(nm[4], nm[5], ks[5], ss[5], ps[5], False, NormLayer, ReLULayer, True))
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
        NormLayer = get_norm_layer(norm_layer)
        self.fc1 = nn.Sequential(nn.Conv2d(nin, nout * 2, kernel_size=1, stride=1, padding=0, bias=True))
        self.fc2 = nn.Sequential(nn.Conv2d(nout, nin, kernel_size=1, stride=1, padding=0, bias=True), NormLayer(nin),
                                 nn.ReLU(True))

    def encoder(self, x):
        res = self.fc1(x)
        mu = res[:, :self.nout, :, :]
        logvar = res[:, self.nout:, :, :]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        return eps.mul(std).add_(mu)

    def decoder(self, x):
        return self.fc2(x)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        c_code = self.reparametrize(mu, logvar)
        res = self.decoder(c_code)
        return mu, logvar, res


class NetI(nn.Module):

    def __init__(self, input_nc, output_dim, norm_layer='batch', relu_layer='leaky', nClasses=2, init_type='normal',
                 init_gain=0.02, gpu_ids=[]):
        super(NetI, self).__init__()
        encoder = Encoder(input_nc, output_dim, norm_layer, relu_layer)
        rec_tail = Rec_Tail(output_dim, nClasses)
        self.encoder = init_net(encoder, init_type, init_gain, gpu_ids)
        self.rec_tail = init_net(rec_tail, init_type, init_gain, gpu_ids)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.avg_pool(feat)
        res = self.rec_tail(feat)
        return feat, res


class NetA(nn.Module):

    def __init__(self, input_nc, output_dim, norm_layer='batch', relu_layer='leaky', vae_dim=100, init_type='normal',
                 init_gain=0.02, gpu_ids=[]):
        super(NetA, self).__init__()
        encoder = Encoder(input_nc, output_dim, norm_layer, relu_layer)
        vae_tail = Vae_Tail(output_dim, vae_dim, norm_layer)
        self.encoder = init_net(encoder, init_type, init_gain, gpu_ids)
        self.vae_tail = init_net(vae_tail, init_type, init_gain, gpu_ids)
        self.ave_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat = self.encoder(x)  ## N*512*1*W -> N*512*1*1
        # n, c, h, w = feat.size()
        global_feat = self.ave_pool(feat)
        mu, logvar, res = self.vae_tail(global_feat)
        # res = res.expand(-1, -1, h, w)
        return mu, logvar, res


class NetG(nn.Module):

    def __init__(self, input_nc, output_nc, norm_layer='batch', relu_layer='relu', init_type='normal', init_gain=0.02,
                 gpu_ids=[]):
        super(NetG, self).__init__()
        NormLayer = get_norm_layer(norm_layer)
        project = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False), NormLayer(512),
                                nn.ReLU(True))
        decoder = Decoder(input_nc, output_nc, norm_layer, relu_layer)
        self.decoder = init_net(decoder, init_type, init_gain, gpu_ids)
        self.project = init_net(project, init_type, init_gain, gpu_ids)

    def forward(self, x1, x2):
        x1_ = self.project(x1)
        return self.decoder(torch.cat((x1_, x2), 1))


class NetD(nn.Module):

    def __init__(self, input_nc, output_dim, norm_layer='batch', relu_layer='leaky', init_type='normal', init_gain=0.02,
                 gpu_ids=[]):
        super(NetD, self).__init__()
        encoder = Encoder(input_nc, output_dim, norm_layer, relu_layer)
        dis_tail = nn.Sequential(nn.Conv2d(output_dim, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.Sigmoid())
        self.encoder = init_net(encoder, init_type, init_gain, gpu_ids)
        self.dis_tail = init_net(dis_tail, init_type, init_gain, gpu_ids)

    def forward(self, x):
        feat = self.encoder(x)
        res = self.dis_tail(feat)
        return feat, res


class GANLoss(nn.Module):

    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCELoss().cuda()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real).cuda()
        return self.loss(input, target_tensor)


class KL_Loss(nn.Module):

    def __init__(self):
        super(KL_Loss, self).__init__()

    def forward(self, mu, logvar):
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.mean(KLD_element).mul_(-0.5)
        return KLD


if __name__ == '__main__':
    netI = NetI(3, 512).cuda()
    netA = NetA(3, 512).cuda()

    input_img = torch.FloatTensor(4, 3, 128, 128).cuda()
    feat_I, res = netI(input_img)
    print('output of I', feat_I.size(), res.size())

    mu, logvar, feat_A = netA(input_img)
    print('output of A', mu.size(), logvar.size(), feat_A.size())

    loss_KL = KL_Loss()(mu, logvar)
    print(loss_KL)

    netG = NetG(1024, 3).cuda()
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
