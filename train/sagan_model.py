# -*- coding: utf-8 -*-

# DCGAN or LSGAN or WGAN or WGAN-GP or DRAGAN, they share the same network architecture

import torch
from torch.autograd import grad
import torch.nn as nn

import config
from train import base_model
import utils
import models


class DCGANModel(base_model.GanModel):

    def train(self):
        print('Training Start')
        for net in [self.netD, self.netG]:
            net.train()
        self.schedulers = [self.schedulersD, self.schedulersG]
        self.loss_names = ['G', 'D_fake', 'D_real']
        if self.args.gp_choice != 'none':
            self.loss_names.append('D_gp')
        self.sampler_count = 0
        self.minibatch_count = 0
        self.epoch_count = 0

        self.loss_iters = []
        self.loss_vis = []

        for epoch in range(self.args.last_epoch, self.args.epochs):
            self.update_learning_rate(self.schedulersD, 'netD')
            self.update_learning_rate(self.schedulersG, 'netG')

            for data in self.train_dataloader:
                self.real = data['img']
                self.batch_size = self.real.size()[0]
                self.epoch_count += 1
                self.minibatch_count += 1
                self.sampler_count += self.batch_size

                self.inputs = torch.rand(self.batch_size, self.args.g_input_dim)

                self.backward_D()
                self.backward_D_gp()
                self.backward_G()

                self.loss_current = [self.loss_G.item(), self.loss_D_fake.item(), self.loss_D_real.item()]
                if self.args.gp_choice != 'none':
                    self.loss_current.append(self.loss_D_gp.item())
                self.loss_iters.append(self.loss_current)

                if self.minibatch_count % self.args.display_fre == 0 or self.sampler_count == self.train_data_size:
                    self.visual_results()

    def backward_D(self):
        self.optD.zero_grad()
        if self.use_gpu:
            self.real = self.real.cuda()
            self.inputs = self.inputs.cuda()
        D_real = self.netD(self.real)[0]
        self.fake = self.netG(self.inputs)[0]
        D_fake = self.netD(self.fake)[0]

        if self.args.loss_choice in ('gan', 'lsgan'):
            self.loss_D_real = self.cri(D_real, True, self.use_gpu)
            self.loss_D_fake = self.cri(D_fake, False, self.use_gpu)
            self.loss_D = self.loss_D_real + self.loss_D_fake
        else:  # wgan under this situation
            self.loss_D_real = -torch.mean(D_real)
            self.loss_D_fake = torch.mean(D_fake)
            self.loss_D = self.loss_D_real + self.loss_D_fake
        self.loss_D.backward(retain_graph=True)
        self.optD.step()

        if self.args.loss_choice == 'wgan' and self.args.gp_choice != 'none':
            for p in self.netD.parameters():
                p.data.clamp(-0.01, 0.01)

    def backward_D_gp(self):
        if self.args.gp_choice == 'wgan-gp':
            alpha = torch.rand(self.batch_size, 1, 1, 1)
            if self.use_gpu:
                alpha = alpha.cuda()
            x_hat = alpha * self.real.data + (1.0 - alpha) * self.fake.data
            x_hat.requires_grad = True
            pred_hat = self.netD(x_hat)[0]
            if self.use_gpu:
                gradients = \
                    grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                         create_graph=True, retain_graph=True)[0]
            else:
                gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                 create_graph=True, retain_graph=True)[0]
            self.loss_D_gp = 10 * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
            self.loss_D_gp.backward(retain_graph=True)
        elif self.args.gp_choice == 'dragan':
            alpha = torch.rand(self.batch_size, 1, 1, 1)
            if self.use_gpu:
                alpha = alpha.cuda()
            if self.use_gpu:
                x_p = self.real + 0.5 * self.real.std() * torch.rand(self.real.size()).cuda()  # 看论文理解公式
            else:
                x_p = self.real + 0.5 * self.real.std() * torch.rand(self.real.size())
            differences = x_p - self.real
            x_hat = self.real + alpha * differences
            x_hat.requires_grad = True
            pred_hat = self.netD(x_hat)[0]
            if self.use_gpu:
                gradients = \
                    grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                         create_graph=True, retain_graph=True)[0]
            else:
                gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                 create_graph=True, retain_graph=True)[0]
            self.loss_D_gp = 10 * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
            self.loss_D_gp.backward(retain_graph=True)
        else:
            pass

    def backward_G(self):
        self.fake = self.netG(self.inputs)[0]
        D_fake = self.netD(self.fake)[0]
        self.optG.zero_grad()
        if self.args.loss_choice in ('gan', 'lsgan'):
            self.loss_G = self.cri(D_fake, True, self.use_gpu)
        else:  # wgan under this situation
            self.loss_G = -torch.mean(D_fake)
        self.loss_G.backward()
        self.optG.step()

    def init_model(self):
        g_norm_layer = utils.nn_utils.get_norm_layer(norm_type=self.args.g_norm_type)
        d_norm_layer = utils.nn_utils.get_norm_layer(norm_type=self.args.d_norm_type)
        self.netG = models.sagan.Generator(image_size=self.args.load_size, conv_dim=64,
                                           z_dim=self.args.g_input_dim)
        self.netD = models.sagan.Discriminator(image_size=self.args.load_size, conv_dim=64)

        utils.nn_utils.print_net(self.netG, 'G')
        utils.nn_utils.print_net(self.netD, 'D')

        if self.use_gpu:
            self.netG = nn.DataParallel(self.netG.cuda())
            self.netD = nn.DataParallel(self.netD.cuda())
        self.optG = torch.optim.Adam(self.netG.parameters(), lr=self.args.g_lr,
                                     betas=(self.args.g_beta1, self.args.g_beta2),
                                     weight_decay=self.args.g_weightdecay)
        self.optD = torch.optim.Adam(self.netD.parameters(), lr=self.args.d_lr,
                                     betas=(self.args.d_beta1, self.args.d_beta2),
                                     weight_decay=self.args.d_weightdecay)
        self.schedulersG = [utils.nn_utils.get_scheduler(opt, self.args.g_lr_policy, self.args.g_lr_decay_step) for
                            opt
                            in
                            [self.optG]]
        self.schedulersD = [utils.nn_utils.get_scheduler(opt, self.args.d_lr_policy, self.args.d_lr_decay_step) for
                            opt
                            in
                            [self.optD]]

        if self.args.loss_choice == 'vanilla':
            self.cri = utils.losses.GANLoss(loss_choice='gan')  # gan
        elif self.args.loss_choice == 'lsgan':
            self.cri = utils.losses.GANLoss(loss_choice='lsgan')  # lsgan
        else:
            self.cri = utils.losses.GANLoss(loss_choice='gan')

    def visual_results(self):
        save_imgs = (self.netG(self.save_inputs)[0].detach().cpu() + 1.0) / 2.0
        self.vis.images(save_imgs, nrow=self.args.save_img_cols, win=1)
        self.loss_vis.append(self.loss_current)
        utils.visual.show_visdom_line(self.vis, self.loss_vis, self.loss_names,
                                      win=2, xlabel='{:d} iters'.format(self.args.display_fre))
        s = 'epoch: {:4d} iters: {:8d}  '.format(
            self.epoch_count, self.minibatch_count)
        for i in range(len(self.loss_names)):
            s += self.loss_names[i] + ': {:.6f}  '.format(self.loss_current[i])
        print(s)


if __name__ == '__main__':
    args = config.train_config()
    model = DCGANModel(args)
    model.train()
