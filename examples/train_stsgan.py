"""
Discriminator:   D
Generator:  G
Conv networks
"""
# region Imports yapf: disable
import argparse
import random
import math
import time
import os
from os.path import basename, normpath
import sys
from collections import deque
from datetime import datetime
import numpy                    as np
import torch
import torch.backends.cudnn     as cudnn
import torch.optim              as optim
import torch.nn                 as nn
from torch.optim.lr_scheduler import StepLR
import torchvision.utils        as vutils
import functools

sys.path.insert(0, os.path.abspath('..'))
import ganbase                  as gb  # pylint: disable=C0413
import itertools

# endregion yapf: enable

# region Arguments yapf: disable

parser = argparse.ArgumentParser()

# region Args for Data
parser.add_argument('--dataset', required=True,
                    help='cifar10 | lsun | imagenet | folder | lfw | lfwcrop | celeba | mnist | cyclegan| stsgan')
parser.add_argument('--dataroot', default=None, help='path to dataset')
parser.add_argument('--nSample', type=int, default=0, help='how many training samples')
parser.add_argument('--loadSize', type=int, default=286, help='the height / width of the image when loading')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nWorkers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--bs', type=int, default=2, help='input batch size')
# endregion

# region Args for Net
parser.add_argument('--n_dim', type=int, default=512)
parser.add_argument('--n_channels', type=int, default=3)
parser.add_argument('--activation', default='leakyrelu',
                    help='leakyrelu | relu | elu | selu | sigmoid | tanh, activation')
parser.add_argument('--normalize', default='batch', help='batch | instance | none, normalization layers')
parser.add_argument('--useDropout', action='store_true', help='if true, use Dropout')

# endregion

# region Args for Training
parser.add_argument('--gpu', type=str, default='0', help='which GPU to use, default to 0')
parser.add_argument('--nIter', type=int, default=50000, help='number of iteration to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for Generator, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam, G. default=0.5')
parser.add_argument('--lambdaI', default=1.0, type=float)
parser.add_argument('--lambdaC', default=1.0, type=float)
parser.add_argument('--lambdaA', default=1.0, type=float)
parser.add_argument('--lambdaGC', default=1.0, type=float)
parser.add_argument('--lambdaGR', default=1.0, type=float)
parser.add_argument('--lambdaGR_alpha', default=0.1, type=float)
parser.add_argument('--lambdaGD', default=1.0, type=float)
parser.add_argument('--lambdaGAN', default=1.0, type=float)
parser.add_argument('--use_no_lsgan', action='store_true', help='if true, use GAN instead of LSGAN')

# endregion

# region Args for Save
parser.add_argument('--nSnapshot', type=int, default=5, help='how many snapshots to keep')
parser.add_argument('--snapshot', default='', help="path to net (to continue training)")
parser.add_argument('--drawIter', type=int, default=500, help='how many iteration per drawing')
parser.add_argument('--nRow', type=int, default=4, help='how many imgs per row')
parser.add_argument('--nCol', type=int, default=2, help='how many imgs per col')
parser.add_argument('--workdir', default=None, help='Where to store samples and models')
# endregion


opt = parser.parse_args()

# endregion yapf: enable

# region Preparation

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
cudnn.benchmark = True

if opt.workdir is None:
    opt.workdir = f'samples/stsgan/exp_{datetime.now()}'.replace(' ', '_')

os.system(f'mkdir -p {opt.workdir}/png')
sys.stdout = gb.Logger(opt.workdir)
print(sys.argv)
print(opt)

opt.manualSeed = random.randint(1, 10000)
print(f"Random Seed: {opt.manualSeed}")
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed(opt.manualSeed)
rng = np.random.RandomState(opt.manualSeed)

snaps = deque([])

# endregion

bs = opt.bs
n_row = opt.nRow
n_col = opt.nCol
valbs = n_row * n_col

dataset, loader, opt.nSample = gb.loaddata(
    opt.dataset, opt.dataroot, opt.imageSize, opt.bs, opt.nSample, opt.nWorkers, droplast=True, loadSize=opt.loadSize)
print(f'{opt.nSample} samples')

# endregion yapf: enable

# region Models
netI = gb.stsgan.NetI(opt.n_channels, opt.n_dim).cuda()
netA = gb.stsgan.NetA(opt.n_channels, opt.n_dim).cuda()
netG = gb.stsgan.NetG(2 * opt.n_dim, opt.n_channels).cuda()
netD = gb.stsgan.NetD(opt.n_channels, opt.n_dim).cuda()
# endregion

# region Optimizers
optimizerI = torch.optim.Adam(netI.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = torch.optim.Adam(itertools.chain(netA.parameters(), netG.parameters()), lr=opt.lr,
                              betas=(opt.beta1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# endregion

iters = 0
d_iter = iter(loader)
timestart = time.time()

cri_Idt = nn.CrossEntropyLoss()
cri_Att = gb.KL_Loss()
cri_GR = nn.MSELoss(reduction='none')
cri_GC = nn.MSELoss()
cri_GD = nn.MSELoss()
cri_GAN = gb.GANLoss(use_lsgan=not opt.use_no_lsgan)

fix_netI = False

assert opt.bs % 2 == 0 and opt.bs >= 4, 'batch size should be mlutiplier of 2 and should be greater than 2'

vis_losses = [0. for i in range(9)]

for it in range(1, opt.nIter - 1):
    optimizerI.zero_grad()
    optimizerD.zero_grad()
    optimizerG.zero_grad()

    # forward
    img, label = next(d_iter)
    Identity = img[0: opt.bs // 2].cuda()
    Attribute = img[opt.bs // 2:].cuda()
    idt_label = label[0: opt.bs // 2].cuda()
    att_label = label[opt.bs // 2:].cuda()
    # backward I
    gb.set_requires_grad(netI, not fix_netI)

    optimizerI.zero_grad()
    feat_I, pred_I = netI(Identity)
    mu, logvar, feat_A = netA(Attribute)
    Result = netG(feat_I, feat_A)

    # TODO unlabeled data
    loss_I = cri_Idt(pred_I.squeeze(3).squeeze(2), idt_label) * opt.lambdaI
    loss_I.backward(retain_graph=True)
    optimizerI.step()

    # backward D
    gb.set_requires_grad(netD, True)
    optimizerD.zero_grad()
    # Real
    _, pred_real = netD(Attribute)
    loss_D_real = cri_GAN(pred_real, True)
    # Fake
    _, pred_fake = netD(Result.detach())
    loss_D_fake = cri_GAN(pred_fake, False)
    # Combined loss
    loss_D = (loss_D_real + loss_D_fake) * 0.5 * opt.lambdaGAN
    loss_D.backward(retain_graph=True)
    optimizerD.step()

    # backward G
    gb.set_requires_grad([netI, netD], False)
    optimizerG.zero_grad()

    loss_A = cri_Att(mu, logvar.detach()) * opt.lambdaA
    loss_GR = (cri_GR(Result, Attribute.detach()).sum(dim=3).sum(dim=2).sum(dim=1) * \
               (opt.lambdaGR_alpha * (idt_label != att_label) + 1. * (
                       idt_label == att_label)).float()).sum() * opt.lambdaGR
    feat_C, pred_C = netI(Result)
    loss_GC = cri_GC(feat_C, feat_I) * opt.lambdaGC

    # TODO ublabeled data
    loss_C = cri_Idt(pred_C.squeeze(3).squeeze(2), idt_label) * opt.lambdaC

    feat_D, pred_D = netD(Result)
    feat_D_, _ = netD(Attribute)
    loss_GAN = cri_GAN(pred_D, True) * opt.lambdaGAN
    loss_GD = cri_GD(feat_D, feat_D_.detach()) * opt.lambdaGD

    loss_G = loss_A + loss_GR + loss_C + loss_GC + loss_GAN + loss_GD
    loss_G.backward()
    optimizerG.step()

    if it % opt.drawIter == 0:
        print(
            'Iter: {:7d} loss_I: {:.3f} loss_A: {:6.3f} loss_GR: {:.3f} loss_C: {:.3f} loss_GC: {:.3f} loss_GD: {:.3f} '
            'lossG: {:.3f} loss_D_fake: {:.3f} loss_D_real: {:.3f}'.format(it, vis_losses[0], vis_losses[1],
                                                                           vis_losses[2], vis_losses[3], vis_losses[4],
                                                                           vis_losses[5], vis_losses[6], vis_losses[7],
                                                                           vis_losses[8]))
        for k in range(len(vis_losses)):
            vis_losses[k] = 0

        state = {'netI': netI.state_dict(), 'netA': netA.state_dict(), 'netG': netG.state_dict(),
                 'netD': netD.state_dict()}
        filename = f'{opt.workdir}/net_iter_{it:06}.pth'
        torch.save(state, filename)
        snaps.append(filename)
        if len(snaps) > opt.nSnapshot:
            os.remove(snaps.popleft())

        # draw in eval mode
        netI.eval()
        netA.eval()
        netG.eval()

        identity = Identity[0: n_row].data
        attribute = Attribute[0: n_row].data
        transform = Result[0: n_row].data
        result = torch.cat((identity, attribute, transform), dim=0)
        vutils.save_image(
            result.mul(0.5).add(0.5),
            f'{opt.workdir}/png/{it:06}_transform.png',
            nrow=n_row)
        # back to train mode
        netI.train()
        netA.train()
        netG.train()

    else:
        vis_losses[0] += loss_I.item()
        vis_losses[1] += loss_A.item()
        vis_losses[2] += loss_GR.item()
        vis_losses[3] += loss_C.item()
        vis_losses[4] += loss_GC.item()
        vis_losses[5] += loss_GD.item()
        vis_losses[6] += loss_G.item()
        vis_losses[7] += loss_D_fake.item()
        vis_losses[8] += loss_D_real.item()

time_used = (time.time() - timestart) / 3600
print(f'time used {time_used:.2} hours')
