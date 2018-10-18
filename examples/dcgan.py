"""
Distance:   D
Generator:  G
Conv networks
"""
#region Imports yapf: disable
import argparse
import random
import math
import time
import os
from os.path                    import basename, normpath
import sys
from collections                import deque
from datetime                   import datetime
import numpy                    as np
import torch
import torch.autograd           as autograd
import torch.backends.cudnn     as cudnn
import torch.optim              as optim
import torch.nn                 as nn
from torch.autograd             import Variable
from torch.optim.lr_scheduler   import StepLR
import torchvision.utils        as vutils
sys.path.insert(0, os.path.abspath('..'))
import ganbase                  as gb # pylint: disable=C0413

#endregion yapf: enable

#region Arguments yapf: disable

parser = argparse.ArgumentParser()

#region Args for Data
parser.add_argument('--dataset',        required=True, help='cifar10 | lsun | imagenet | folder | lfw | lfwcrop | celeba | mnist')
parser.add_argument('--dataroot',       default=None, help='path to dataset')
parser.add_argument('--nSample',        type=int, default=0, help='how many training samples')
parser.add_argument('--imageSize',      type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc',             type=int, default=3, help='input image channels')
parser.add_argument('--nWorkers',       type=int, default=2, help='number of data loading workers')
parser.add_argument('--nz',             type=int, default=64, help='intrinsic dim of latent space')
parser.add_argument('--bs',             type=int, default=256, help='input batch size')
#endregion

#region Args for Net
parser.add_argument('--widthG',         type=int, default=64, help='init width of dcgan G')
parser.add_argument('--widthD',         type=int, default=64, help='init width of dcgan D')
parser.add_argument('--nExtraLayerG',   type=int, default=0, help='extra layer of dcgan G')
parser.add_argument('--nExtraLayerD',   type=int, default=0, help='extra layer of dcgan D')
parser.add_argument('--nExtraConvG',    type=int, default=0, help='extra conv of dcgan G')
parser.add_argument('--nExtraConvD',    type=int, default=0, help='extra conv of dcgan D')
parser.add_argument('--activationG',    default='leakyrelu', help='leakyrelu | relu | elu | selu | sigmoid | tanh, activation for G')
parser.add_argument('--activationD',    default='leakyrelu', help='leakyrelu | relu | elu | selu | sigmoid | tanh, activation for D')
parser.add_argument('--normalizeG',     default='batch', help='batch | instance | none, normalization layers for G')
parser.add_argument('--normalizeD',     default='batch', help='batch | instance | none, normalization layers for D')
#endregion

#region Args for Training
parser.add_argument('--gpu',            type=int, default=0, help='which GPU to use, default to 0')
parser.add_argument('--nIter',          type=int, default=50000, help='number of iteration to train for')
parser.add_argument('--repeatD',        type=int, default=1, help='repeat D per iteration')
parser.add_argument('--repeatG',        type=int, default=1, help='repeat G per iteration')
parser.add_argument('--optimizerG',     default='adam', help='adam | rmsprop | sgd, optimizer for G')
parser.add_argument('--optimizerD',     default='adam', help='adam | rmsprop | sgd, optimizer for D')
parser.add_argument('--lrG',            type=float, default=0.0001, help='learning rate for Generator, default=0.0001')
parser.add_argument('--lrD',            type=float, default=0.0001, help='learning rate for Distance, default=0.0001')
parser.add_argument('--beta1G',         type=float, default=0.5, help='beta1 for adam, G. default=0.5')
parser.add_argument('--beta1D',         type=float, default=0.5, help='beta1 for adam, D. default=0.5')
parser.add_argument('--momentG',        type=float, default=0.9, help='moment for sgd, G. default=0.5')
parser.add_argument('--momentD',        type=float, default=0.9, help='moment for sgd, D. default=0.5')
#endregion

#region Args for Save
parser.add_argument('--nSnapshot',      type=int, default=5, help='how many snapshots to keep')
parser.add_argument('--snapshotG',      default='', help="path to net G (to continue training)")
parser.add_argument('--snapshotD',      default='', help="path to net D (to continue training)")
parser.add_argument('--drawIter',       type=int, default=500, help='how many epoch per drawing')
parser.add_argument('--nRow',           type=int, default=10, help='how many imgs per row')
parser.add_argument('--nCol',           type=int, default=10, help='how many imgs per col')
parser.add_argument('--workdir',        default=None, help='Where to store samples and models')
#endregion

opt = parser.parse_args()

#endregion yapf: enable

#region Preparation

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
cudnn.benchmark = True

if opt.workdir is None:
    opt.workdir = f'samples/exp73_{datetime.now()}'.replace(' ', '_')

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

#endregion

#region Parameters yapf: disable

latent    = gb.GaussLatent(opt.nz)
dim_z     = opt.nz
bs        = opt.bs
n_row     = opt.nRow
n_col     = opt.nCol
valbs     = n_row * n_col
z_draw_np = latent.sample(valbs).float()
z_draw    = z_draw_np.cuda(non_blocking=True)

dataset, loader, opt.nSample = gb.loaddata(
    opt.dataset, opt.dataroot, opt.imageSize, opt.bs, opt.nSample, opt.nWorkers, droplast=True)
print(f'{opt.nSample} samples')

#endregion yapf: enable

#region Models

# model G
netG = gb.DCGAN_G(opt.imageSize, opt.nc, opt.nz, opt.widthG, opt.nExtraLayerG,
                  opt.nExtraConvG, opt.activationG, opt.normalizeG)
netG.apply(gb.weights_init)

if opt.snapshotG != '':
    netG.load_state_dict(torch.load(opt.snapshotG))

netG = netG.cuda()
print(netG)

# model D
netD = gb.DCGAN_D(opt.imageSize, opt.nc, opt.widthD, opt.nExtraLayerD,
                  opt.nExtraConvD, opt.activationD, opt.normalizeD, "sigmoid",
                  1)
netD.apply(gb.weights_init)

if opt.snapshotD != '':
    netD.load_state_dict(torch.load(opt.snapshotD))

netD = netD.cuda()
print(netD)

# optimizers
if opt.optimizerG == 'adam':
    optimizerG = optim.Adam(
        netG.parameters(), lr=opt.lrG, betas=(opt.beta1G, 0.9))
elif opt.optimizerG == 'rmsprop':
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)
elif opt.optimizerG == 'sgd':
    optimizerG = optim.SGD(netG.parameters(), lr=opt.lrG, momentum=opt.momentG)
else:
    raise ValueError('optimizer not supported')

if opt.optimizerD == 'adam':
    optimizerD = optim.Adam(
        netD.parameters(), lr=opt.lrD, betas=(opt.beta1D, 0.9))
elif opt.optimizerD == 'rmsprop':
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
elif opt.optimizerD == 'sgd':
    optimizerD = optim.SGD(netD.parameters(), lr=opt.lrD, momentum=opt.momentD)
else:
    raise ValueError('optimizer not supported')

#endregion

#region Training

iters = 0
d_iter = iter(loader)
timestart = time.time()
ones_label = torch.ones(bs, 1).cuda()
zeros_label = torch.zeros(bs, 1).cuda()
loss = nn.BCELoss()

prob_D_real, prob_D_fake, prob_G = 0., 0., 0.
for it in range(1, opt.nIter - 1):

    ############################
    # Update Distance D
    ############################
    #region D
    for p in netD.parameters():
        p.requires_grad = True
    for p in netG.parameters():
        p.requires_grad = False

    for r in range(opt.repeatD):
        netD.zero_grad()

        x_cpu, _ = next(d_iter)
        x_real = Variable(x_cpu).cuda(non_blocking=True)
        z_np = latent.sample(opt.bs).float()
        with torch.no_grad():
            z = z_np.cuda(non_blocking=True)
            x_fake = netG(z)
        x_fake = Variable(x_fake.data)

        loss_fake = loss(netD(x_fake), Variable(zeros_label))
        loss_real = loss(netD(x_real), Variable(ones_label))

        loss_fake.backward()
        loss_real.backward()

        prob_D_real += np.exp(-loss_real.data.item())
        prob_D_fake += 1 - np.exp(-loss_fake.data.item())

        optimizerD.step()
    #endregion

    ############################
    # Update Generator G
    ############################
    #region G
    for p in netD.parameters():
        p.requires_grad = False
    for p in netG.parameters():
        p.requires_grad = True

    for r in range(opt.repeatG):
        netG.zero_grad()

        z_np = latent.sample(opt.bs).float()
        z = z_np.cuda(non_blocking=True)

        loss_gen = loss(netD(netG(z)), Variable(ones_label))
        loss_gen.backward()

        prob_G += np.exp(-loss_real.data.item())

        optimizerG.step()
    #endregion

    if it % opt.drawIter == 0 or it == 1:
        prob_D_real /= opt.drawIter
        prob_D_fake /= opt.drawIter
        prob_G /= opt.drawIter

        print(
            f'{datetime.now()}[{it}/{opt.nIter}] probability for D real/fake {prob_D_real:.5}/{prob_D_fake:.5}, G {prob_G:.5}'
        )

        # eval mode for drawing
        netG.eval()

        # 1. fixed random fake
        fake = netG(Variable(z_draw))
        vutils.save_image(
            fake.data.mul(0.5).add(0.5),
            f'{opt.workdir}/png/{it:06}.png',
            nrow=n_row)

        # 2. random fake
        z_rand_np = latent.sample(valbs).float()
        z_rand = z_rand_np.cuda(non_blocking=True)
        fake = netG(Variable(z_rand))
        vutils.save_image(
            fake.data.mul(0.5).add(0.5),
            f'{opt.workdir}/png/{it:06}_rand.png',
            nrow=n_row)

        # back to train mode
        netG.train()

        prob_D_real = 0
        prob_D_fake = 0
        prob_G = 0
        #endregion

        #region Checkpoint

        filename = f'{opt.workdir}/netG_epoch_{it:06}.pth'
        torch.save(netG.state_dict(), filename)
        snaps.append(filename)
        filename = f'{opt.workdir}/netD_epoch_{it:06}.pth'
        torch.save(netD.state_dict(), filename)
        snaps.append(filename)
        if len(snaps) > 2 * opt.nSnapshot:
            os.remove(snaps.popleft())
            os.remove(snaps.popleft())

    #endregion

time_used = (time.time() - timestart) / 3600
print(f'time used {time_used:.2} hours')

#endregion
