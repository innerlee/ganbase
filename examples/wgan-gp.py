"""
Discriminator:   D
Generator:  G
Conv networks
"""
#region Imports yapf: disable
import argparse
import random
import time
import os
import sys
from collections                import deque
from datetime                   import datetime
import numpy                    as np
import torch
from torch.autograd import grad
import torchvision.utils        as vutils
sys.path.insert(0, os.path.abspath('..'))
from use_logger import use_logger
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
parser.add_argument('--normalizeD',     default='instance', help='batch | instance | none, normalization layers for D')
#endregion

#region Args for Training
parser.add_argument('--gpu',            type=int, default=0, help='which GPU to use, default to 0')
parser.add_argument('--nIter',          type=int, default=50000, help='number of iteration to train for')
parser.add_argument('--repeatD',        type=int, default=5, help='repeat D per iteration')
parser.add_argument('--repeatG',        type=int, default=1, help='repeat G per iteration')
parser.add_argument('--optimizerG',     default='adam', help='adam | rmsprop | sgd, optimizer for G')
parser.add_argument('--optimizerD',     default='adam', help='adam | rmsprop | sgd, optimizer for D')
parser.add_argument('--lrG',            type=float, default=0.0001, help='learning rate for Generator, default=0.0001')
parser.add_argument('--lrD',            type=float, default=0.0001, help='learning rate for Discriminator, default=0.0001')
parser.add_argument('--beta1G',         type=float, default=0, help='beta1 for adam, G. default=0')
parser.add_argument('--beta1D',         type=float, default=0, help='beta1 for adam, D. default=0')
parser.add_argument('--momentG',        type=float, default=0.9, help='moment for sgd, G. default=0.9')
parser.add_argument('--momentD',        type=float, default=0.9, help='moment for sgd, D. default=0.9')
parser.add_argument('--lambdaGP',       type=float, default=10, help='weight for gradient penalty, default=10')

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

gb.visible_gpu(opt.gpu)

if opt.workdir is None:
    opt.workdir = f'samples/wgangp_{opt.dataset}/exp_{datetime.now()}'.replace(' ', '_')
use_logger(opt.workdir)
print(sys.argv)
print(opt)

gb.random_seed()
saver = gb.Saver(slot=2, keepnum=3)

#endregion

#region Parameters yapf: disable

latent    = gb.GaussLatent(opt.nz)
z_draw_np = latent.sample_gauss(opt.nRow * opt.nCol).float()
z_draw    = z_draw_np.cuda(non_blocking=True)

dataset, loader, opt.nSample = gb.loaddata(
    opt.dataset, opt.dataroot, opt.imageSize, opt.bs, opt.nSample, opt.nWorkers, droplast=True)
print(f'{opt.nSample} samples')

#endregion yapf: enable

#region Models
startIter = 1

# model G
netG = gb.DCGAN_G(opt.imageSize, opt.nc, opt.nz, opt.widthG, opt.nExtraLayerG,
                  opt.nExtraConvG, opt.activationG, opt.normalizeG)
netG.apply(gb.weights_init)

if opt.snapshotG != '':
   netG, startIter = saver.load(netG, opt.snapshotG)

netG = netG.cuda()
print(netG)

# model D
netD = gb.DCGAN_D(opt.imageSize, opt.nc, opt.widthD, opt.nExtraLayerD,
                  opt.nExtraConvD, opt.activationD, opt.normalizeD, "none",
                  1)
netD.apply(gb.weights_init)

if opt.snapshotD != '':
    netD, startIter = saver.load(netD, opt.snapshotD)
netD = netD.cuda()
print(netD)

# optimizers

optimizerG = gb.get_optimizer(netG.parameters(), opt.optimizerG, lr=opt.lrG, beta1=opt.beta1G, beta2=0.999, eps=1e-8,
                              weight_decay=0, alpha=0.99, momentum=opt.momentG, centered=False, dampening=0,
                              nesterov=False)
optimizerD = gb.get_optimizer(netD.parameters(), opt.optimizerD, lr=opt.lrD, beta1=opt.beta1D, beta2=0.999, eps=1e-8,
                              weight_decay=0, alpha=0.99, momentum=opt.momentD, centered=False, dampening=0,
                              nesterov=False)

#endregion

#region Training

iters = 0
d_iter = iter(loader)
timestart = time.time()

loss_D_real, loss_D_fake, loss_D_gp, loss_G = 0., 0., 0., 0.
for it in range(startIter, opt.nIter - 1):

    ############################
    # Update Discriminator D
    ############################
    #region D
    for p in netD.parameters():
        p.requires_grad = True
    for p in netG.parameters():
        p.requires_grad = False

    for r in range(opt.repeatD):
        netD.zero_grad()

        x_cpu, _ = next(d_iter)
        x_real = x_cpu.cuda(non_blocking=True)
        z_np = latent.sample_gauss(opt.bs).float()
        with torch.no_grad():
            z = z_np.cuda(non_blocking=True)
            x_fake = netG(z)
        x_fake = x_fake.data

        loss_fake = torch.mean(netD(x_fake))
        loss_real = -torch.mean(netD(x_real))

        batch_size = x_real.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).cuda()
        x_hat = alpha * x_real.data + (1.0 - alpha) * x_fake.data
        x_hat.requires_grad = True
        x_hat = x_hat.cuda()
        pred_hat = netD(x_hat)

        gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]

        loss_gp = opt.lambdaGP * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

        loss_fake.backward()
        loss_real.backward()
        loss_gp.backward(retain_graph=True)

        loss_D_real += loss_real.data.item()
        loss_D_fake += loss_fake.data.item()
        loss_D_gp += loss_gp.data.item()

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

        z_np = latent.sample_gauss(opt.bs).float()
        z = z_np.cuda(non_blocking=True)

        loss_gen = -torch.mean(netD(netG(z)))
        loss_gen.backward()

        loss_G += loss_gen.data.item()

        optimizerG.step()
    #endregion

    if it % opt.drawIter == 0 or it == 1:
        loss_D_real /= opt.drawIter
        loss_D_fake /= opt.drawIter
        loss_G /= opt.drawIter

        print(
            f'{datetime.now()}[{it}/{opt.nIter}] loss for D real/fake/gp {loss_D_real:.5}/{loss_D_fake:.5}/{loss_D_gp:.5}, G {loss_G:.5}'
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
        z_rand_np = latent.sample_gauss(opt.nRow * opt.nCol).float()
        z_rand = z_rand_np.cuda(non_blocking=True)
        fake = netG(z_rand)
        vutils.save_image(
            fake.data.mul(0.5).add(0.5),
            f'{opt.workdir}/png/{it:06}_rand.png',
            nrow=opt.nRow)

        # back to train mode
        netG.train()

        loss_D_real = 0
        loss_D_fake = 0
        loss_D_gp = 0
        loss_G = 0
        #endregion

        #region Checkpoint

        filename = f'{opt.workdir}/netG_epoch_{it:06}.pth'
        saver.save(netG.state_dict(), filename, it)
        filename = f'{opt.workdir}/netD_epoch_{it:06}.pth'
        saver.save(netD.state_dict(), filename, it)

    #endregion

time_used = (time.time() - timestart) / 3600
print(f'time used {time_used:.2} hours')

#endregion
