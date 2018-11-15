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
import torch.autograd           as autograd
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
                    help='cifar10 | lsun | imagenet | folder | lfw | lfwcrop | celeba | mnist | cyclegan')
parser.add_argument('--dataroot', default=None, help='path to dataset')
parser.add_argument('--datatarget', default=None, help='path to target dataset')
parser.add_argument('--nSample', type=int, default=0, help='how many training samples')
parser.add_argument('--loadSize', type=int, default=72, help='the height / width of the image when loading')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nWorkers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--bs', type=int, default=256, help='input batch size')
# endregion

# region Args for Net
parser.add_argument('--widthG', type=int, default=64, help='init width of cyclegan G')
parser.add_argument('--widthD', type=int, default=64, help='init width of cyclegan D')
parser.add_argument('--nDownSamplingG', type=int, default=2, help='log(2, n_downsamlping) in cyclegan G')
parser.add_argument('--nExtraLayerG', type=int, default=6, help='resnet blocks in cyclegan G')
parser.add_argument('--nExtraLayerD', type=int, default=3, help='pathGAN choice of cyclegan D')
parser.add_argument('--nExtraConvG', type=int, default=0, help='extra conv of cyclegan G')
parser.add_argument('--nExtraConvD', type=int, default=0, help='extra conv of cyclegan D')
parser.add_argument('--activationG', default='leakyrelu',
                    help='leakyrelu | relu | elu | selu | sigmoid | tanh, activation for G')
parser.add_argument('--activationD', default='leakyrelu',
                    help='leakyrelu | relu | elu | selu | sigmoid | tanh, activation for D')
parser.add_argument('--normalizeG', default='batch', help='batch | instance | none, normalization layers for G')
parser.add_argument('--normalizeD', default='batch', help='batch | instance | none, normalization layers for D')
parser.add_argument('--useDropoutG', action='store_true', help='if true, use Dropout in G')

# endregion

# region Args for Training
parser.add_argument('--gpu', type=str, default=0, help='which GPU to use, default to 0')
parser.add_argument('--nIter', type=int, default=50000, help='number of iteration to train for')
parser.add_argument('--repeatD', type=int, default=1, help='repeat D per iteration')
parser.add_argument('--repeatG', type=int, default=1, help='repeat G per iteration')
parser.add_argument('--optimizerG', default='adam', help='adam | rmsprop | sgd, optimizer for G')
parser.add_argument('--optimizerD', default='adam', help='adam | rmsprop | sgd, optimizer for D')
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate for Generator, default=0.0001')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate for Discriminator, default=0.0001')
parser.add_argument('--beta1G', type=float, default=0.5, help='beta1 for adam, G. default=0.5')
parser.add_argument('--beta1D', type=float, default=0.5, help='beta1 for adam, D. default=0.5')
parser.add_argument('--momentG', type=float, default=0.9, help='moment for sgd, G. default=0.5')
parser.add_argument('--momentD', type=float, default=0.9, help='moment for sgd, D. default=0.5')
parser.add_argument('--lambdaA', default=10.0, type=float, help='lambda for A')
parser.add_argument('--lambdaB', default=10.0, type=float, help='lambda for B')
parser.add_argument('--lambdaI', default=0.5, type=float, help='lambda for Idt')

# endregion

# region Args for Save
parser.add_argument('--nSnapshot', type=int, default=5, help='how many snapshots to keep')
parser.add_argument('--snapshotG_A', default='', help="path to net G_A (to continue training)")
parser.add_argument('--snapshotG_B', default='', help="path to net G_B (to continue training)")
parser.add_argument('--snapshotD_A', default='', help="path to net D_A (to continue training)")
parser.add_argument('--snapshotD_B', default='', help="path to net D_B (to continue training)")
parser.add_argument('--drawIter', type=int, default=500, help='how many epoch per drawing')
parser.add_argument('--nRow', type=int, default=10, help='how many imgs per row')
parser.add_argument('--nCol', type=int, default=10, help='how many imgs per col')
parser.add_argument('--workdir', default=None, help='Where to store samples and models')
# endregion


opt = parser.parse_args()

# endregion yapf: enable

# region Preparation

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
cudnn.benchmark = True

if opt.workdir is None:
    opt.workdir = f'samples/cyclegan/exp_{datetime.now()}'.replace(' ', '_')

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
    opt.dataset, opt.dataroot, opt.imageSize, opt.bs, opt.nSample, opt.nWorkers, droplast=True, loadSize=opt.loadSize,
    datatarget=opt.datatarget)
print(f'{opt.nSample} samples')

# endregion yapf: enable

# region Models

# model G


# TOOD adjust model

netG_A = gb.cyclegan.CycleGAN_G(opt.imageSize, opt.nc, opt.widthG, opt.nDownSamplingG, opt.nExtraLayerG,
                                activation=opt.activationG,
                                normalize=opt.normalizeG, use_dropout=opt.useDropoutG)
netG_A.apply(gb.weights_init)

if opt.snapshotG_A != '':
    netG_A.load_state_dict(torch.load(opt.snapshotG_A))

netG_A = netG_A.cuda()
print(netG_A)

netG_B = gb.cyclegan.CycleGAN_G(opt.imageSize, opt.nc, opt.widthG, opt.nDownSamplingG, opt.nExtraLayerG,
                                activation=opt.activationG,
                                normalize=opt.normalizeG, use_dropout=opt.useDropoutG)
netG_B.apply(gb.weights_init)

if opt.snapshotG_B != '':
    netG_B.load_state_dict(torch.load(opt.snapshotG_B))

netG_B = netG_B.cuda()
print(netG_B)

# model D

netD_A = gb.cyclegan.CycleGAN_D(opt.imageSize, opt.nc, opt.widthD, opt.nExtraLayerD, activation=opt.activationD,
                                normalize=opt.normalizeD, outactivation='none', outdim=1)
netD_A.apply(gb.weights_init)

if opt.snapshotD_A != '':
    netD_A.load_state_dict(torch.load(opt.snapshotD_A))

netD_A = netD_A.cuda()
print(netD_A)

netD_B = gb.cyclegan.CycleGAN_D(opt.imageSize, opt.nc, opt.widthD, opt.nExtraLayerD, activation=opt.activationD,
                                normalize=opt.normalizeD, outactivation='none', outdim=1)
netD_B.apply(gb.weights_init)

if opt.snapshotD_B != '':
    netD_B.load_state_dict(torch.load(opt.snapshotD_B))

netD_B = netD_B.cuda()
print(netD_B)

# optimizers
if opt.optimizerG == 'adam':
    optimizerG = optim.Adam(
        itertools.chain(netG_A.parameters(), netG_B.parameters()), lr=opt.lrG, betas=(opt.beta1G, 0.9))
elif opt.optimizerD == 'rmsprop':
    optimizerD = optim.RMSprop(itertools.chain(netG_A.parameters(), netG_B.parameters()), lr=opt.lrG)
elif opt.optimizerD == 'sgd':
    optimizerD = optim.SGD(itertools.chain(netG_A.parameters(), netG_B.parameters()), lr=opt.lrG, momentum=opt.momentG)
else:
    raise ValueError('optimizer not supported')

if opt.optimizerD == 'adam':
    optimizerD = optim.Adam(
        itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=opt.lrD, betas=(opt.beta1D, 0.9))
elif opt.optimizerD == 'rmsprop':
    optimizerD = optim.RMSprop(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=opt.lrD)
elif opt.optimizerD == 'sgd':
    optimizerD = optim.SGD(itertools.chain(netD_A.parameters(), netD_B.parameters()), lr=opt.lrD, momentum=opt.momentD)
else:
    raise ValueError('optimizer not supported')

# endregion

iters = 0
d_iter = iter(loader)
timestart = time.time()
cri_GAN = gb.GANLoss(use_lsgan=True)
cri_Cycle = nn.L1Loss()
cri_Idt = nn.L1Loss()

prob_D_A_real, prob_D_A_fake, prob_G_A = 0., 0., 0.
prob_D_B_real, prob_D_B_fake, prob_G_B = 0., 0., 0.

for it in range(1, opt.nIter - 1):

    ############################
    # Update Generator G
    ############################
    # region G

    for r in range(opt.repeatG):
        gb.set_requires_grad([netD_A, netD_B], False)
        gb.set_requires_grad([netG_A, netG_B], True)
        optimizerG.zero_grad()

        real_A, real_B, _, _ = next(d_iter)
        real_A = real_A.cuda()
        real_B = real_B.cuda()

        fake_B = netG_A(real_A)
        idt_A = netG_A(real_B)
        rec_A = netG_B(fake_B)
        fake_A = netG_B(real_B)
        idt_B = netG_B(real_A)
        rec_B = netG_A(fake_A)

        loss_idt_A = cri_Idt(idt_A, real_B) * opt.lambdaA * opt.lambdaI
        loss_idt_B = cri_Idt(idt_B, real_A) * opt.lambdaB * opt.lambdaI
        loss_G_A = cri_GAN(netD_A(fake_B), True)
        loss_G_B = cri_GAN(netD_B(fake_A), True)
        loss_cycle_A = cri_Cycle(rec_A, real_A) * opt.lambdaA
        loss_cycle_B = cri_Cycle(rec_B, real_B) * opt.lambdaB
        loss_gen = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_gen.backward(retain_graph=True)
        optimizerG.step()
        prob_G_A += np.exp(-loss_G_A.detach().item())
        prob_G_A += np.exp(-loss_G_B.detach().item())

    # endregion

    ############################
    # Update Discriminator D
    ############################
    # region D

    for r in range(opt.repeatD):
        gb.set_requires_grad([netD_A, netD_B], True)
        gb.set_requires_grad([netG_A, netG_B], False)
        optimizerD.zero_grad()
        loss_D_A_real = cri_GAN(netD_A(real_B), False)
        loss_D_A_fake = cri_GAN(netD_A(fake_B), True)
        loss_D_A = (loss_D_A_fake + loss_D_A_real) / 2.
        loss_D_B_real = cri_GAN(netD_B(fake_A), False)
        loss_D_B_fake = cri_GAN(netD_B(real_A), True)
        loss_D_B = (loss_D_A_real + loss_D_B_fake) / 2
        loss_D_A.backward(retain_graph=True)
        loss_D_B.backward(retain_graph=True)

        prob_D_A_real += np.exp(-loss_D_A_real.detach().item())
        prob_D_A_fake += 1 - np.exp(-loss_D_A_fake.detach().item())
        prob_D_B_real += np.exp(-loss_D_B_real.detach().item())
        prob_D_B_fake += 1 - np.exp(-loss_D_B_fake.detach().item())

        optimizerD.step()
    # endregion
    if it % opt.drawIter == 0 or it == 1:
        prob_D_A_real /= (opt.drawIter * opt.repeatD)
        prob_D_A_fake /= (opt.drawIter * opt.repeatD)
        prob_D_B_real /= (opt.drawIter * opt.repeatD)
        prob_D_B_fake /= (opt.drawIter * opt.repeatD)
        prob_G_A /= (opt.drawIter * opt.repeatG)
        prob_G_B /= (opt.drawIter * opt.repeatG)

        print(
            f'{datetime.now()}[{it}/{opt.nIter}] probability for '
            f'D_A real/fake {prob_D_A_real:.5}/{prob_D_A_fake:.5},'
            f'D_B real/fake {prob_D_B_real:.5}/{prob_D_B_fake:.5},'
            f' G_A {prob_G_A:.5}'
            f' G_B {prob_G_B:.5}'
        )
        # eval mode for drawing
        netG_A.eval()
        netG_B.eval()

        # TODO style transfer to the fixed images
        # # 1. fixed random fake
        # fake = netG(Variable(z_draw))
        # vutils.save_image(
        #     fake.data.mul(0.5).add(0.5),
        #     f'{opt.workdir}/png/{it:06}.png',
        #     nrow=n_row)
        #
        # fake = netG(Variable(z_rand))
        # vutils.save_image(
        #     fake.data.mul(0.5).add(0.5),
        #     f'{opt.workdir}/png/{it:06}_rand.png',
        #     nrow=n_row)

        # back to train mode
        netG_A.train()
        netG_B.train()
        # endregion

        # region Checkpoint

        filename = f'{opt.workdir}/netG_A_epoch_{it:06}.pth'
        torch.save(netG_A.state_dict(), filename)
        snaps.append(filename)
        filename = f'{opt.workdir}/netG_B_epoch_{it:06}.pth'
        torch.save(netG_B.state_dict(), filename)
        snaps.append(filename)
        filename = f'{opt.workdir}/netD_A_epoch_{it:06}.pth'
        torch.save(netD_A.state_dict(), filename)
        snaps.append(filename)
        filename = f'{opt.workdir}/netD_B_epoch_{it:06}.pth'
        torch.save(netD_B.state_dict(), filename)
        snaps.append(filename)
        if len(snaps) > 4 * opt.nSnapshot:
            for _ in range(4):
                os.remove(snaps.popleft())

    # endregion

time_used = (time.time() - timestart) / 3600
print(f'time used {time_used:.2} hours')

# endregion
