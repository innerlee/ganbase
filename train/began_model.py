# -*- coding: utf-8 -*-

# DCGAN or LSGAN or WGAN or WGAN-GP or DRAGAN, they share the same network architecture

import torch
from torch.autograd import grad
from collections import deque
import numpy as np
import torch.nn as nn

import config
from train import base_model
import utils
import models


class BEGANModel(base_model.GanModel):

    def train(self):
        print('Training Start')
        for net in [self.netD, self.netG]:
            net.train()
        self.loss_names = ['G', 'D_fake', 'D_real']
        if self.args.gp_choice != 'none':
            self.loss_names.append('D_gp')
        self.sampler_count = 0
        self.minibatch_count = 0
        self.epoch_count = 0

        self.loss_iters = []
        self.loss_vis = []

        self.k_t = 0.
        self.prev_measure = 1
        self.measure_history = deque([0] * self.args.lr_update_step, self.args.lr_update_step)

        for epoch in range(self.args.last_epoch, self.args.epochs):

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

                g_d_balance = (self.args.gamma * self.loss_D_real - self.loss_D_fake).detach().item()
                self.k_t += self.args.lambda_k * g_d_balance
                self.k_t = max(min(1, self.k_t), 0)

                measure = self.loss_D_real.detach().item() + abs(g_d_balance)
                self.measure_history.append(measure)

                if self.minibatch_count % self.args.lr_update_step == 0:
                    cur_measure = np.mean(self.measure_history)
                    if cur_measure > prev_measure * 0.9999:
                        prev_measure = cur_measure
                        for param_group in self.optG.param_groups:
                            param_group['lr'] *= 0.5
                        for param_group in self.optD.param_groups:
                            param_group['lr'] *= 0.5

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
        D_real = self.netD(self.real)
        self.fake = self.netG(self.inputs)
        D_fake = self.netD(self.fake)

        self.loss_D_real = torch.mean(torch.abs(D_real - self.real))
        self.loss_D_fake = torch.mean(torch.abs(D_fake - self.fake))
        self.loss_D = self.loss_D_real - self.k_t * self.loss_D_fake

        self.loss_D.backward(retain_graph=True)
        self.optD.step()

    def backward_D_gp(self):
        if self.args.gp_choice == 'wgan-gp':
            alpha = torch.rand(self.batch_size, 1, 1, 1)
            if self.use_gpu:
                alpha = alpha.cuda()
            x_hat = alpha * self.real.data + (1.0 - alpha) * self.fake.data
            x_hat.requires_grad = True
            pred_hat = self.netD(x_hat)
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
            pred_hat = self.netD(x_hat)
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
        self.fake = self.netG(self.inputs)
        D_fake = self.netD(self.fake)
        self.optG.zero_grad()
        self.loss_G = torch.mean(D_fake - self.fake)
        self.loss_G.backward()
        self.optG.step()

    def init_model(self):
        g_norm_layer = utils.nn_utils.get_norm_layer(norm_type=self.args.g_norm_type)
        d_norm_layer = utils.nn_utils.get_norm_layer(norm_type=self.args.d_norm_type)

        self.netG = models.began.Generator(z_size=self.args.g_input_dim, img_size=self.args.load_size,
                                           init_square_size=8, img_channel=3)

        self.netD = models.began.Discriminator(img_size=self.args.load_size, z_size=self.args.g_input_dim,
                                               init_square_size=8, img_channel=3)

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


if __name__ == '__main__':
    args = config.train_config()
    model = BEGANModel(args)
    model.train()
