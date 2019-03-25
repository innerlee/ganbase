# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import pickle
from torchvision.utils import save_image

import config
from train import base_model
import models
import utils


class Pix2PixModel(base_model.GanModel):

    def get_dataloader(self):
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        train_dataset = utils.datasets.Pix2PixDataset(self.args.train_dir, self.args, mean, std)
        train_data_size = len(train_dataset)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.train_batch,
            shuffle=True,
            num_workers=self.args.workers
        )
        return train_data_size, train_loader

    def init_model(self):
        g_norm_layer = utils.nn_utils.get_norm_layer(norm_type=self.args.g_norm_type)
        d_norm_layer = utils.nn_utils.get_norm_layer(norm_type=self.args.d_norm_type)
        self.netG = models.pix2pix_cyclegan.UnetGenerator(self.args.g_input_dim, self.args.g_output_dim,
                                                          self.args.g_unet_depth, self.args.g_net_width,
                                                          norm_layer=g_norm_layer, use_dropout=self.args.use_dropout)

        self.netD = models.pix2pix_cyclegan.NLayerDiscriminator(self.args.d_input_dim, self.args.d_net_width,
                                                                self.args.d_num_layers, norm_layer=d_norm_layer)

        utils.nn_utils.print_net(self.netG, 'G')
        utils.nn_utils.print_net(self.netD, 'D')

        if self.use_gpu:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            self.netG = nn.DataParallel(self.netG.cuda())
            self.netD = nn.DataParallel(self.netD.cuda())
        self.optG = torch.optim.Adam(self.netG.parameters(),
                                     lr=self.args.g_lr, betas=(self.args.g_beta1, self.args.g_beta2),
                                     weight_decay=self.args.g_weightdecay)
        self.optD = torch.optim.Adam(self.netD.parameters(),
                                     lr=self.args.d_lr, betas=(self.args.d_beta1, self.args.d_beta2),
                                     weight_decay=self.args.d_weightdecay)
        self.schedulersG = [utils.nn_utils.get_scheduler(opt, self.args.g_lr_policy, self.args.g_lr_decay_step) for opt
                            in [self.optG]]
        self.schedulersD = [utils.nn_utils.get_scheduler(opt, self.args.d_lr_policy, self.args.d_lr_decay_step) for opt
                            in [self.optD]]

        if self.args.loss_choice == 'vanilla':
            self.cri_GAN = utils.losses.GANLoss(loss_choice='vanilla')
        else:  # lsgan
            self.cri_GAN = utils.losses.GANLoss(loss_choice='lsgan')
        self.cri_Style = nn.L1Loss()

    def train(self):
        print('Training Start')
        for net in [self.netD, self.netG]:
            net.train()
        self.schedulers = [self.schedulersD, self.schedulersG]
        self.loss_names = ['D_real', 'D_fake', 'G_GAN', 'G_Style']
        self.sampler_count = 0
        self.minibatch_count = 0
        self.epoch_count = 0
        self.loss_iters = []
        self.loss_vis = []

        for epoch in range(self.args.last_epoch, self.args.epochs):
            self.update_learning_rate(self.schedulersD, 'netD')
            self.update_learning_rate(self.schedulersG, 'netG')
            self.epoch_count += 1
            for data in self.train_dataloader:
                self.real_A = data['A']
                self.real_B = data['B']
                self.batch_size = self.real_A.size()[0]
                self.minibatch_count += 1
                self.sampler_count += self.batch_size
                if self.use_gpu:
                    self.real_A = self.real_A.cuda()
                    self.real_B = self.real_B.cuda()

                self.fake_B = self.netG(self.real_A)

                self.backward_G()
                self.backward_D()

                self.loss_current = [
                    self.loss_D_real.item(),
                    self.loss_D_fake.item(),
                    self.loss_G_GAN.item(),
                    self.loss_G_Style.item()
                ]

                self.loss_iters.append(self.loss_current)

                if self.minibatch_count % self.args.display_fre == 0 or self.sampler_count == self.train_data_size:
                    self.visual_results()
            self.save_model()

    def backward_D(self):
        self.optD.zero_grad()
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        self.loss_D_real = self.cri_GAN(self.netD(real_AB), True, use_cuda=self.use_gpu)
        self.loss_D_fake = self.cri_GAN(self.netD(fake_AB), False, use_cuda=self.use_gpu)
        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        self.loss_D.backward()
        self.optD.step()

    def backward_G(self):
        self.optG.zero_grad()
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        self.loss_G_GAN = self.cri_GAN(self.netD(fake_AB), True, use_cuda=self.use_gpu)
        self.loss_G_Style = self.cri_Style(self.fake_B, self.real_B) * self.args.lambdaI
        self.loss_G = self.loss_G_GAN + self.loss_G_Style
        self.loss_G.backward(retain_graph=True)
        self.optG.step()

    def load_model(self):
        if self.args.last_epoch > 0:
            state = torch.load(os.path.join(self.save_dir, str(self.args.last_epoch).zfill(4) + '_state.pth'))
            if self.use_gpu:
                self.netG.module.load_state_dict(state['netG'])
                self.netD.module.load_state_dict(state['netD'])
            else:
                self.netG.load_state_dict(state['netG'])
                self.netD.load_state_dict(state['netD'])
            self.optG.load_state_dict(state['optG'])
            self.optD.load_state_dict(state['optD'])
            print('Load model of epoch {:4d}'.format(self.args.last_epoch))
        else:
            # TODO add init type and init gain
            utils.nn_utils.init_weights(self.netG, 'G', init_type=self.args.init_type, std=self.args.init_std)
            utils.nn_utils.init_weights(self.netD, 'D', init_type=self.args.init_type, std=self.args.init_std)

    def save_model(self):
        state = {
            'epoch': self.epoch_count,
            'netG': self.netG.module.state_dict() if self.use_gpu else self.netG.state_dict(),
            'netD': self.netD.module.state_dict() if self.use_gpu else self.netD.state_dict(),
            'optG': self.optG.state_dict(),
            'optD': self.optD.state_dict()
        }
        if (self.epoch_count) % self.args.save_fre == 0:
            torch.save(state, os.path.join(self.args.save_dir, str(self.epoch_count).zfill(4) + '_state.pth'))

        torch.save(state, os.path.join(self.args.save_dir, 'latest_state.pth'))
        with open(os.path.join(self.args.save_dir, 'loss_iters.pkl'), 'wb') as f:
            pickle.dump(self.loss_iters, f)

        save_imgs = self.vis_imgs
        save_image(save_imgs, os.path.join(self.save_img_dir, 'epoch_{:s}.jpg'.format(str(self.epoch_count).zfill(4))),
                   nrow=self.args.save_img_cols)

    def visual_results(self):
        self.vis_imgs = [
            self.real_A[0, :, :, :].cpu().detach(),
            self.fake_B[0, :, :, :].cpu().detach(),
            self.real_B[0, :, :, :].cpu().detach()
        ]
        self.vis_imgs = torch.stack(self.vis_imgs)
        self.vis_imgs = (self.vis_imgs + 1.0) / 2.0
        self.vis.images(self.vis_imgs, nrow=4, win=1)
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
    model = Pix2PixModel(args)
    model.train()
