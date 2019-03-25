# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import itertools
import os
import pickle
from torchvision.utils import save_image

import config
from train import base_model
import models
import utils


class AttentionCycleGANModel(base_model.GanModel):

    def get_dataloader(self):
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        train_transform = utils.datasets.get_pil_transform('train', self.args.resize_choice, self.args.load_size,
                                                           self.args.fine_size, self.args.flip, mean, std)

        train_dataset = utils.datasets.UnalignedDataset2(data_dir_A=self.args.train_dir,
                                                         data_dir_B=self.args.train_target_dir,
                                                         transform=train_transform)
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
        a_norm_layer = utils.nn_utils.get_norm_layer(norm_type=self.args.a_norm_type)
        self.netG_A = models.attention_cyclegan.ResnetGenerator(self.args.g_input_dim, self.args.g_output_dim,
                                                                self.args.g_net_width, norm_layer=g_norm_layer,
                                                                use_dropout=self.args.use_dropout,
                                                                n_blocks=self.args.g_resnet_blocks)
        self.netG_B = models.attention_cyclegan.ResnetGenerator(self.args.g_input_dim, self.args.g_output_dim,
                                                                self.args.g_net_width, norm_layer=g_norm_layer,
                                                                use_dropout=self.args.use_dropout,
                                                                n_blocks=self.args.g_resnet_blocks)
        self.netD_A = models.attention_cyclegan.NLayerDiscriminator(self.args.d_input_dim, self.args.d_net_width,
                                                                    self.args.d_num_layers, norm_layer=d_norm_layer)
        self.netD_B = models.attention_cyclegan.NLayerDiscriminator(self.args.d_input_dim, self.args.d_net_width,
                                                                    self.args.d_num_layers, norm_layer=d_norm_layer)
        self.netA_A = models.attention_cyclegan.AttentionNet(self.args.g_input_dim, 1, self.args.a_net_width,
                                                             a_norm_layer)
        self.netA_B = models.attention_cyclegan.AttentionNet(self.args.g_input_dim, 1, self.args.a_net_width,
                                                             a_norm_layer)

        utils.nn_utils.print_net(self.netG_A, 'G_A')
        utils.nn_utils.print_net(self.netG_B, 'G_B')
        utils.nn_utils.print_net(self.netD_A, 'D_A')
        utils.nn_utils.print_net(self.netD_B, 'D_B')
        utils.nn_utils.print_net(self.netA_B, 'A_A')
        utils.nn_utils.print_net(self.netA_B, 'A_B')

        if self.use_gpu:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

            self.netG_A = nn.DataParallel(self.netG_A.cuda())
            self.netG_B = nn.DataParallel(self.netG_B.cuda())
            self.netD_A = nn.DataParallel(self.netD_A.cuda())
            self.netD_B = nn.DataParallel(self.netD_B.cuda())
            self.netA_A = nn.DataParallel(self.netA_A.cuda())
            self.netA_B = nn.DataParallel(self.netA_B.cuda())

        self.optG = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                     lr=self.args.g_lr, betas=(self.args.g_beta1, self.args.g_beta2),
                                     weight_decay=self.args.g_weightdecay)
        self.optD = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                     lr=self.args.d_lr, betas=(self.args.d_beta1, self.args.d_beta2),
                                     weight_decay=self.args.d_weightdecay)
        self.optA = torch.optim.Adam(itertools.chain(self.netA_A.parameters(), self.netA_B.parameters()),
                                     lr=self.args.a_lr, betas=(self.args.a_beta1, self.args.a_beta2),
                                     weight_decay=self.args.a_weightdecay)
        self.schedulersG = [utils.nn_utils.get_scheduler(opt, self.args.g_lr_policy, self.args.g_lr_decay_step,
                                                         linear_steps=self.args.epochs // 2) for opt
                            in [self.optG]]
        self.schedulersD = [utils.nn_utils.get_scheduler(opt, self.args.d_lr_policy, self.args.d_lr_decay_step,
                                                         linear_steps=self.args.epochs // 2) for opt
                            in [self.optD]]
        self.schedulersA = [utils.nn_utils.get_scheduler(opt, self.args.a_lr_policy, self.args.a_lr_decay_step,
                                                         linear_steps=self.args.epochs // 2) for opt
                            in [self.optA]]

        if self.args.loss_choice == 'vanilla':
            self.cri_GAN = utils.losses.GANLoss(loss_choice='gan')
        else:  # lsgan
            self.cri_GAN = utils.losses.GANLoss(loss_choice='lsgan')
        self.cri_Cycle = nn.L1Loss()
        self.cri_Idt = nn.L1Loss()

    def train(self):
        print('Training Start')
        for net in [self.netD_A, self.netD_B, self.netG_A, self.netG_B]:
            net.train()
        self.schedulers = [self.schedulersD, self.schedulersG]
        self.loss_names = ['D_A_real', 'D_A_fake', 'G_A', 'D_B_real', 'D_B_fake', 'G_B',
                           'cycle_A', 'cycle_B', 'idt_A', 'idt_B']
        self.sampler_count = 0
        self.minibatch_count = 0
        self.epoch_count = 0
        self.loss_iters = []
        self.loss_vis = []

        for epoch in range(self.args.last_epoch, self.args.epochs):
            self.update_learning_rate(self.schedulersG, 'netG')
            self.update_learning_rate(self.schedulersD, 'netD')
            self.update_learning_rate(self.schedulersA, 'netA')
            self.epoch_count += 1

            if self.epoch_count == self.args.train_att_epochs + 1:
                self.remove_instance_norm()

            if self.epoch_count >= self.args.train_att_epochs + 1:
                utils.nn_utils.set_requires_grad([self.netA_A, self.netA_B], False)

            for data in self.train_dataloader:
                self.real_A = data['A']
                self.real_B = data['B']
                self.batch_size = self.real_A.size()[0]
                self.minibatch_count += 1
                self.sampler_count += self.batch_size

                if self.use_gpu:
                    self.real_A = self.real_A.cuda()
                    self.real_B = self.real_B.cuda()

                self.mask_real_A = self.netA_A(self.real_A).repeat(1, self.args.g_output_dim, 1, 1)
                self.fake_B = self.netG_A(self.real_A).mul(self.mask_real_A) + self.real_A.mul(
                    1. - self.mask_real_A)

                self.mask_fake_B = self.netA_B(self.fake_B).repeat(1, self.args.g_output_dim, 1, 1)
                self.rec_A = self.netG_B(self.fake_B).mul(self.mask_fake_B) + self.fake_B.mul(1 - self.mask_fake_B)

                self.idt_A = self.netG_B(self.real_A)

                self.mask_real_B = self.netA_B(self.real_B).repeat(1, self.args.g_output_dim, 1, 1)
                self.fake_A = self.netG_B(self.real_B).mul(self.mask_real_B) + self.real_B.mul(
                    1. - self.mask_real_B)

                self.mask_fake_A = self.netA_A(self.fake_A).repeat(1, self.args.g_output_dim, 1, 1)
                self.rec_B = self.netG_A(self.fake_A).mul(self.mask_fake_A) + self.fake_A.mul(1. - self.mask_fake_A)

                self.idt_B = self.netG_A(self.real_B)

                if self.epoch_count > self.args.train_att_epochs:
                    self.fake_B_new = torch.where(self.mask_real_A > self.args.mask_tau, self.netG_A(self.real_A),
                                                  torch.Tensor([0]))
                    self.real_B_new = torch.where(self.mask_real_B > self.args.mask_tau, self.real_B,
                                                  torch.Tensor([0]))
                    self.fake_A_new = torch.where(self.mask_real_B > self.args.mask_tau, self.netG_B(self.real_B),
                                                  torch.Tensor([0]))
                    self.real_A_new = torch.where(self.mask_real_A > self.args.mask_tau, self.real_A,
                                                  torch.Tensor([0]))

                self.backward_G_A()
                self.backward_D()

                self.loss_current = [
                    self.loss_D_A_real.item(),
                    self.loss_D_A_fake.item(),
                    self.loss_G_A.item(),
                    self.loss_D_B_real.item(),
                    self.loss_D_B_fake.item(),
                    self.loss_G_B.item(),
                    self.loss_cycle_A.item(),
                    self.loss_cycle_B.item(),
                    self.loss_idt_A.item(),
                    self.loss_idt_B.item()
                ]

                self.loss_iters.append(self.loss_current)

                if self.minibatch_count % self.args.display_fre == 0 or self.sampler_count == self.train_data_size:
                    self.visual_results()
            self.save_model()

    def backward_D(self):
        self.fake_A_pool = utils.nn_utils.ImagePool(50)
        self.fake_B_pool = utils.nn_utils.ImagePool(50)
        self.fake_B = self.fake_B_pool.query(self.fake_B)
        self.fake_A = self.fake_A_pool.query(self.fake_A)

        utils.nn_utils.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optD.zero_grad()
        # netA 目的是生成B风格的图片，netB 目的是生成A风格的图片
        if self.epoch_count <= self.args.train_att_epochs + 1:
            self.loss_D_A_real = self.cri_GAN(self.netD_A(self.real_B), True, use_cuda=self.use_gpu)
            self.loss_D_A_fake = self.cri_GAN(self.netD_A(self.fake_B), False, use_cuda=self.use_gpu)
            self.loss_D_B_real = self.cri_GAN(self.netD_B(self.real_A), True, use_cuda=self.use_gpu)
            self.loss_D_B_fake = self.cri_GAN(self.netD_B(self.fake_A), False, use_cuda=self.use_gpu)
        else:
            self.loss_D_A_real = self.cri_GAN(self.netD_A(self.real_B_new), True, use_cuda=self.use_gpu)
            self.loss_D_A_fake = self.cri_GAN(self.netD_A(self.fake_B_new), False, use_cuda=self.use_gpu)
            self.loss_D_B_real = self.cri_GAN(self.netD_B(self.real_A_new), True, use_cuda=self.use_gpu)
            self.loss_D_B_fake = self.cri_GAN(self.netD_B(self.fake_A_new), False, use_cuda=self.use_gpu)
        self.loss_D_A = (self.loss_D_A_real + self.loss_D_A_fake) / 2
        self.loss_D_B = (self.loss_D_B_real + self.loss_D_B_fake) / 2
        self.loss_D_A.backward()
        self.loss_D_B.backward()
        self.optD.step()

    def backward_G_A(self):
        utils.nn_utils.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optG.zero_grad()
        if self.epoch_count <= self.args.train_att_epochs + 1:
            self.optA.zero_grad()

        self.loss_idt_A = self.cri_Idt(self.idt_A, self.real_B) * self.args.lambdaA * self.args.lambdaI
        self.loss_idt_B = self.cri_Idt(self.idt_B, self.real_A) * self.args.lambdaB * self.args.lambdaI
        if self.epoch_count <= self.args.train_att_epochs + 1:
            self.loss_G_A = self.cri_GAN(self.netD_A(self.fake_B), True, use_cuda=self.use_gpu)
            self.loss_G_B = self.cri_GAN(self.netD_B(self.fake_A), True, use_cuda=self.use_gpu)
        else:
            self.loss_G_A = self.cri_GAN(self.netD_A(self.fake_B_new), True, use_cuda=self.use_gpu)
            self.loss_G_B = self.cri_GAN(self.netD_B(self.fake_A_new), True, use_cuda=self.use_gpu)
        self.loss_cycle_A = self.cri_Cycle(self.rec_A, self.real_A) * self.args.lambdaA
        self.loss_cycle_B = self.cri_Cycle(self.rec_B, self.real_B) * self.args.lambdaB
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + \
                      self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

        self.loss_G.backward(retain_graph=True)
        self.optG.step()
        if self.epoch_count <= self.args.train_att_epochs + 1:
            self.optA.step()

    def remove_instance_norm(self):
        # remove instance norm after some epochs as paper
        # new_netG_A = models.attention_cyclegan.ResnetGenerator(self.args.g_input_dim, self.args.g_output_dim,
        #                                                        self.args.g_net_width, norm_layer=None,
        #                                                        with_norm=False,
        #                                                        use_dropout=self.args.use_dropout,
        #                                                        n_blocks=self.args.g_resnet_blocks)
        # new_netG_B = models.attention_cyclegan.ResnetGenerator(self.args.g_input_dim, self.args.g_output_dim,
        #                                                        self.args.g_net_width, norm_layer=None,
        #                                                        with_norm=False,
        #                                                        use_dropout=self.args.use_dropout,
        #                                                        n_blocks=self.args.g_resnet_blocks)
        new_netD_A = models.attention_cyclegan.NLayerDiscriminator(self.args.d_input_dim,
                                                                   self.args.d_net_width,
                                                                   self.args.d_num_layers,
                                                                   norm_layer=None)
        new_netD_B = models.attention_cyclegan.NLayerDiscriminator(self.args.d_input_dim,
                                                                   self.args.d_net_width,
                                                                   self.args.d_num_layers,
                                                                   norm_layer=None)
        if self.use_gpu:
            new_netD_A = nn.DataParallel(new_netD_A.cuda())
            new_netD_B = nn.DataParallel(new_netD_B.cuda())
            # new_netG_A = nn.DataParallel(new_netG_A.cuda())
            # new_netG_B = nn.DataParallel(new_netG_B.cuda())
        if self.args.load_when_switch:
            utils.nn_utils.load_pretrained_dict_ordered(self.netD_A, new_netD_A)
            utils.nn_utils.load_pretrained_dict_ordered(self.netD_B, new_netD_B)
            # utils.nn_utils.load_pretrained_dict_ordered(self.netG_A, new_netG_A)
            # utils.nn_utils.load_pretrained_dict_ordered(self.netG_B, new_netG_B)
        else:
            # utils.nn_utils.init_weights(self.netG_A, 'new_G_A', init_type=self.args.init_type,
            #                             std=self.args.init_std)
            # utils.nn_utils.init_weights(self.netG_B, 'new_G_B', init_type=self.args.init_type,
            #                             std=self.args.init_std)
            utils.nn_utils.init_weights(self.netD_A, 'new_D_A', init_type=self.args.init_type,
                                        std=self.args.init_std)
            utils.nn_utils.init_weights(self.netD_B, 'new_D_B', init_type=self.args.init_type,
                                        std=self.args.init_std)

        self.netD_A = new_netD_A
        self.netD_B = new_netD_B
        #TODO to test whether load optimizer state will help
        self.optD = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                     lr=self.args.d_lr, betas=(self.args.d_beta1, self.args.d_beta2),
                                     weight_decay=self.args.d_weightdecay)
        # self.netG_A = new_netG_A
        # self.netG_B = new_netG_B

    def load_model(self):
        if self.args.last_epoch > 0:
            state = torch.load(os.path.join(self.save_dir, str(self.args.last_epoch).zfill(4) + '_state.pth'))
            if self.use_gpu:
                self.netG_A.module.load_state_dict(state['netG_A'])
                self.netG_B.module.load_state_dict(state['netG_B'])
                self.netD_A.module.load_state_dict(state['netD_A'])
                self.netD_B.module.load_state_dict(state['netD_B'])
                self.netA_A.module.load_state_dict(state['netA_A'])
                self.netA_B.module.load_state_dict(state['netA_B'])
            else:
                self.netG_A.load_state_dict(state['netG_A'])
                self.netG_B.load_state_dict(state['netG_B'])
                self.netD_A.load_state_dict(state['netD_A'])
                self.netD_B.load_state_dict(state['netD_B'])
                self.netA_A.load_state_dict(state['netA_A'])
                self.netA_B.load_state_dict(state['netA_B'])
            self.optG.load_state_dict(state['optG'])
            self.optD.load_state_dict(state['optD'])
            self.optA.load_state_dict(state['optA'])
            print('Load model of epoch {:4d}'.format(self.args.last_epoch))
        else:
            # TODO add init type and init gain
            utils.nn_utils.init_weights(self.netG_A, 'G_A', init_type=self.args.init_type, std=self.args.init_std)
            utils.nn_utils.init_weights(self.netG_B, 'G_B', init_type=self.args.init_type, std=self.args.init_std)
            utils.nn_utils.init_weights(self.netD_A, 'D_A', init_type=self.args.init_type, std=self.args.init_std)
            utils.nn_utils.init_weights(self.netD_B, 'D_B', init_type=self.args.init_type, std=self.args.init_std)
            utils.nn_utils.init_weights(self.netA_A, 'A_A', init_type=self.args.init_type, std=self.args.init_std)
            utils.nn_utils.init_weights(self.netA_B, 'A_B', init_type=self.args.init_type, std=self.args.init_std)

    def save_model(self):
        state = {
            'epoch': self.epoch_count,
            'netG_A': self.netG_A.module.state_dict() if self.use_gpu else self.netG_A.state_dict(),
            'netG_B': self.netG_B.module.state_dict() if self.use_gpu else self.netG_B.state_dict(),
            'netD_A': self.netD_A.module.state_dict() if self.use_gpu else self.netD_A.state_dict(),
            'netD_B': self.netD_B.module.state_dict() if self.use_gpu else self.netD_B.state_dict(),
            'netA_A': self.netA_A.module.state_dict() if self.use_gpu else self.netA_A.state_dict(),
            'netA_B': self.netA_B.module.state_dict() if self.use_gpu else self.netA_B.state_dict(),
            'optG': self.optG.state_dict(),
            'optD': self.optD.state_dict(),
            'optA': self.optA.state_dict()
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
            self.mask_real_A[0, :, :, :].cpu().detach() * 2. - 1.,
            self.fake_B[0, :, :, :].cpu().detach(),
            self.mask_fake_B[0, :, :, :].cpu().detach() * 2. - 1.,
            self.rec_A[0, :, :, :].cpu().detach(),
            self.idt_A[0, :, :, :].cpu().detach(),
            self.real_B[0, :, :, :].cpu().detach(),
            self.mask_real_B[0, :, :, :].cpu().detach() * 2. - 1.,
            self.fake_A[0, :, :, :].cpu().detach(),
            self.mask_fake_A[0, :, :, :].cpu().detach() * 2. - 1.,
            self.rec_B[0, :, :, :].cpu().detach(),
            self.idt_B[0, :, :, :].cpu().detach()
        ]
        self.vis_imgs = torch.stack(self.vis_imgs)
        self.vis_imgs = (self.vis_imgs + 1.0) / 2.0
        self.vis.images(self.vis_imgs, nrow=6, win=1)
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
    model = AttentionCycleGANModel(args)
    model.train()
