"""
Discriminator:   D
Generator:  G
Conv networks
"""
# region Imports yapf: disable
import argparse
import random
import time
import os
import sys
from collections import deque
from datetime import datetime
import numpy                    as np
import torch
import torch.backends.cudnn     as cudnn
import torch.optim              as optim
import torch.nn                 as nn
import torchvision.utils        as vutils
from use_logger import use_logger

sys.path.insert(0, os.path.abspath('..'))
import ganbase                  as gb  # pylint: disable=C0413

# endregion yapf: enable

# region Arguments yapf: disable

parser = argparse.ArgumentParser()

# region Args for Data
parser.add_argument('--dataset', required=True,
                    help='cifar10 | lsun | imagenet | folder | lfw | lfwcrop | celeba | mnist')
parser.add_argument('--dataroot', default=None, help='path to dataset')
parser.add_argument('--nSample', type=int, default=0, help='how many training samples')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nWorkers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--nz', type=int, default=64, help='intrinsic dim of latent space')
parser.add_argument('--bs', type=int, default=16, help='input batch size')
# endregion

# region Args for Net
parser.add_argument('--width', type=int, default=128, help='n in the paper')
parser.add_argument('--activationG', default='elu',
                    help='leakyrelu | relu | elu | selu | sigmoid | tanh, activation for G')
parser.add_argument('--activationD', default='elu',
                    help='leakyrelu | relu | elu | selu | sigmoid | tanh, activation for D')
parser.add_argument('--normalizeG', default='none', help='batch | instance | none, normalization layers for G')
parser.add_argument('--normalizeD', default='none', help='batch | instance | none, normalization layers for D')
# endregion

# region Args for Training
parser.add_argument('--gpu', type=int, default=0, help='which GPU to use, default to 0')
parser.add_argument('--nIter', type=int, default=100000, help='number of iteration to train for')
parser.add_argument('--repeatD', type=int, default=1, help='repeat D per iteration')
parser.add_argument('--repeatG', type=int, default=1, help='repeat G per iteration')
parser.add_argument('--optimizerG', default='adam', help='adam | rmsprop | sgd, optimizer for G')
parser.add_argument('--optimizerD', default='adam', help='adam | rmsprop | sgd, optimizer for D')
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate for Generator, default=0.0001')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate for Discriminator, default=0.0001')
parser.add_argument('--lr_update_step', type=int, default=3000)
parser.add_argument('--beta1G', type=float, default=0.5, help='beta1 for adam, G. default=0.5')
parser.add_argument('--beta1D', type=float, default=0.5, help='beta1 for adam, D. default=0.5')
parser.add_argument('--momentG', type=float, default=0.9, help='moment for sgd, G. default=0.5')
parser.add_argument('--momentD', type=float, default=0.9, help='moment for sgd, D. default=0.5')
parser.add_argument('--gamma', type=float, default=0.5, help='default=0.5')
parser.add_argument('--lambda_k', type=float, default=0.001, help='the learning rate for k, default=0.001')
# endregion

# region Args for Save
parser.add_argument('--nSnapshot', type=int, default=5, help='how many snapshots to keep')
parser.add_argument('--snapshotG', default='', help="path to net G (to continue training)")
parser.add_argument('--snapshotD', default='', help="path to net D (to continue training)")
parser.add_argument('--drawIter', type=int, default=500, help='how many epoch per drawing')
parser.add_argument('--nRow', type=int, default=10, help='how many imgs per row')
parser.add_argument('--nCol', type=int, default=10, help='how many imgs per col')
parser.add_argument('--workdir', default=None, help='Where to store samples and models')
# endregion

opt = parser.parse_args()

# endregion yapf: enable

# region Preparation

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
cudnn.benchmark = True

if opt.workdir is None:
    opt.workdir = f'samples/began_{opt.dataset}/exp_{datetime.now()}'.replace(' ', '_')
use_logger(opt.workdir)

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

# region Parameters yapf: disable

latent = gb.GaussLatent(opt.nz)
z_draw_np = latent.sample_uniform(opt.nRow * opt.nCol).float()
z_draw = z_draw_np.cuda(non_blocking=True)

dataset, loader, opt.nSample = gb.loaddata(
    opt.dataset, opt.dataroot, opt.imageSize, opt.bs, opt.nSample, opt.nWorkers, droplast=True)
print(f'{opt.nSample} samples')

# endregion yapf: enable

# region Models

# model G
netG = gb.BEGAN_G(opt.imageSize, opt.nc, opt.nz, opt.width, opt.activationG, opt.normalizeG)
#netG = gb.Decoder()
#netG.apply(gb.weights_init)


if opt.snapshotG != '':
    netG.load_state_dict(torch.load(opt.snapshotG))

netG = nn.DataParallel(netG.cuda())
print(netG)

# model D
netD = gb.BEGAN_D(opt.imageSize, opt.nc, opt.nz, opt.width, opt.activationD, opt.normalizeD)
#netD = gb.Discriminator()

if opt.snapshotD != '':
    netD.load_state_dict(torch.load(opt.snapshotD))

netD = nn.DataParallel(netD.cuda())
print(netD)


# optimizers

if opt.optimizerG == 'adam':
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1G, 0.999))
elif opt.optimizerG == 'rmsprop':
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)
elif opt.optimizerG == 'sgd':
    optimizerG = optim.SGD(netG.parameters(), lr=opt.lrG, momentum=opt.momentG)
else:
    raise ValueError('optimizer not supported')



if opt.optimizerD == 'adam':
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1D, 0.999))
elif opt.optimizerD == 'rmsprop':
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
elif opt.optimizerD == 'sgd':
    optimizerD = optim.SGD(netD.parameters(), lr=opt.lrD, momentum=opt.momentD)
else:
    raise ValueError('optimizer not supported')



# endregion

# region Training

iters = 0
d_iter = iter(loader)
timestart = time.time()

loss_D, loss_G = 0., 0.

k_t = 0
prev_measure = 1
measure_history = deque([0] * opt.lr_update_step, opt.lr_update_step)

for it in range(1, opt.nIter - 1):


    x_cpu, _ = next(d_iter)
    x_real = x_cpu.cuda()
    #print(x_real.size())
    if opt.dataset=='mnist':
        x_real = torch.cat(tensors=[x_real,x_real,x_real],dim=1)
    #print(x_real.size())

    netD.zero_grad()

    z = latent.sample_uniform(opt.bs).float()
    z = z.cuda()

    x_fake = netG(z)

    d_loss_real = torch.mean(torch.abs(netD(x_real)-x_real))
    d_loss_fake = torch.mean(torch.abs(netD(x_fake.detach())- x_fake))

    d_loss = d_loss_real - k_t * d_loss_fake
    d_loss.backward()
    optimizerD.step()


    netG.zero_grad()
    x_fake = netG(z)

    g_loss = torch.mean(torch.abs(netD(x_fake) - x_fake))
    g_loss.backward()
    optimizerG.step()

    loss_D += d_loss.data.item()
    loss_G += g_loss.data.item()


    g_d_balance = (opt.gamma * d_loss_real - d_loss_fake).detach().item()
    k_t += opt.lambda_k * g_d_balance
    k_t = max(min(1, k_t), 0)

    measure = d_loss_real.detach().item() + abs(g_d_balance)
    measure_history.append(measure)

    if it % opt.drawIter == 0 or it == 1:
        if it!=1:
            loss_D /= opt.drawIter
            loss_G /= opt.drawIter

        print(
            f'{datetime.now()}[{it}/{opt.nIter}] loss for D {loss_D:.5}, G {loss_G:.5}  measure {measure: .5} lr {opt.lrG: .5}'
        )

        # eval mode for drawing
        netG.eval()

        # 1. fixed random fake
        fake = netG(z_draw)
        vutils.save_image(
            fake.data.mul(0.5).add(0.5),
            f'{opt.workdir}/png/{it:06}.png',
            nrow=opt.nRow)

        # 2. random fake
        z_rand_np = latent.sample_uniform(opt.nRow * opt.nCol).float()
        z_rand = z_rand_np.cuda(non_blocking=True)
        fake = netG(z_rand)
        vutils.save_image(
            fake.data.mul(0.5).add(0.5),
            f'{opt.workdir}/png/{it:06}_rand.png',
            nrow=opt.nRow)

        # back to train mode
        netG.train()

        loss_G = 0
        loss_D = 0
        # endregion

        # region Checkpoint

        filename = f'{opt.workdir}/netG_epoch_{it:06}.pth'
        torch.save(netG.state_dict(), filename)
        snaps.append(filename)
        filename = f'{opt.workdir}/netD_epoch_{it:06}.pth'
        torch.save(netD.state_dict(), filename)
        snaps.append(filename)
        if len(snaps) > 2 * opt.nSnapshot:
            os.remove(snaps.popleft())
            os.remove(snaps.popleft())

    if it % opt.lr_update_step == opt.lr_update_step - 1:

        cur_measure = np.mean(measure_history)
        if cur_measure > prev_measure * 0.9999:
            opt.lrG *= 0.5
            opt.lrD *= 0.5
            prev_measure = cur_measure
            for param_group in optimizerG.param_groups:
                param_group['lr'] = opt.lrG
            for param_group in optimizerD.param_groups:
                param_group['lr'] = opt.lrD


time_used = (time.time() - timestart) / 3600
print(f'time used {time_used:.2} hours')

# endregion
