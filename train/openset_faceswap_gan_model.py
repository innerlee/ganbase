# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import pickle
from torchvision.utils import save_image
import torchvision
import itertools

import config
from train import base_model
import models
import utils


class OpensetFaceSwapGANModel(base_model.GanModel):

    def get_dataloader(self):
        transform_list = []
        if self.args.resize_choice >= 1:
            transform_list.append(transforms.Resize(self.args.load_size))
        if self.args.resize_choice == 2:
            transform_list.append(transforms.CenterCrop(self.args.fine_size))
        elif self.args.resize_choice == 3:
            transform_list.append(transforms.RandomCrop(self.args.fine_size))
        if self.args.flip == 1:
            transform_list.append(transforms.RandomHorizontalFlip(0.5))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

        transform = transforms.Compose(transform_list)

        # TODO make better dataset
        # train_dataset = utils.datasets.AlignedDataset2(data_dir_A=self.args.train_dir,
        #                                                data_dir_B=self.args.train_target_dir,
        #                                                transform=transform)
        train_dataset = torchvision.datasets.ImageFolder(self.args.train_dir, transform=transform)

        train_data_size = len(train_dataset)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.train_batch,
            shuffle=True,
            num_workers=self.args.workers,
            drop_last=True
        )
        return train_data_size, train_loader

    def init_model(self):
        i_norm_layer = utils.nn_utils.get_norm_layer(norm_type=self.args.ia_norm_type)
        a_norm_layer = utils.nn_utils.get_norm_layer(norm_type=self.args.ia_norm_type)
        g_norm_layer = utils.nn_utils.get_norm_layer(norm_type=self.args.g_norm_type)
        d_norm_layer = utils.nn_utils.get_norm_layer(norm_type=self.args.d_norm_type)
        self.netI = models.openset_faceswap_gan.NetI(self.args.g_input_dim, self.args.ia_output_dim, i_norm_layer,
                                                     nClasses=self.args.n_classes)
        self.netA = models.openset_faceswap_gan.NetA(self.args.g_input_dim, self.args.ia_output_dim, a_norm_layer)
        self.netG = models.openset_faceswap_gan.NetG(2 * self.args.ia_output_dim, self.args.g_output_dim, g_norm_layer)
        self.netD = models.openset_faceswap_gan.NetD(self.args.d_input_dim, self.args.ia_output_dim, d_norm_layer)

        utils.nn_utils.print_net(self.netI, 'I')
        utils.nn_utils.print_net(self.netA, 'A')
        utils.nn_utils.print_net(self.netG, 'G')
        utils.nn_utils.print_net(self.netD, 'D')

        if self.use_gpu:
            self.netI = nn.DataParallel(self.netI.cuda())
            self.netA = nn.DataParallel(self.netA.cuda())
            self.netG = nn.DataParallel(self.netG.cuda())
            self.netD = nn.DataParallel(self.netD.cuda())

        self.optI = torch.optim.Adam(self.netI.parameters(),
                                     lr=self.args.i_lr, betas=(self.args.i_beta1, self.args.i_beta2),
                                     weight_decay=self.args.i_weightdecay)
        self.optG = torch.optim.Adam(itertools.chain(self.netA.parameters(), self.netG.parameters()),
                                     lr=self.args.g_lr, betas=(self.args.g_beta1, self.args.g_beta2),
                                     weight_decay=self.args.g_weightdecay)
        self.optD = torch.optim.Adam(self.netD.parameters(),
                                     lr=self.args.d_lr, betas=(self.args.d_beta1, self.args.d_beta2),
                                     weight_decay=self.args.d_weightdecay)
        self.schedulersI = [utils.nn_utils.get_scheduler(opt, self.args.i_lr_policy, self.args.i_lr_decay_step) for opt
                            in [self.optI]]
        self.schedulersG = [utils.nn_utils.get_scheduler(opt, self.args.g_lr_policy, self.args.g_lr_decay_step) for opt
                            in [self.optG]]
        self.schedulersD = [utils.nn_utils.get_scheduler(opt, self.args.d_lr_policy, self.args.d_lr_decay_step) for opt
                            in [self.optD]]

        if self.args.loss_choice == 'vanilla':
            self.cri_GAN = utils.losses.GANLoss(loss_choice='vanilla')
        else:  # lsgan
            self.cri_GAN = utils.losses.GANLoss(loss_choice='lsgan')

        self.cri_Idt = nn.CrossEntropyLoss()
        self.cri_Att = utils.losses.KL_Loss()
        self.cri_GR = nn.MSELoss()
        # self.cri_GR = nn.MSELoss(reduction='none')
        self.cri_GC = nn.MSELoss()
        self.cri_GD = nn.MSELoss()

    def train(self):
        print('Training Start')
        for net in [self.netD, self.netG, self.netA, self.netI]:
            net.train()
        self.schedulers = [self.schedulersD, self.schedulersG, self.schedulersI]
        self.loss_names = ['I', 'A', 'GR', 'C', 'GC', 'GD', 'G', 'D_fake', 'D_real']
        self.sampler_count = 0
        self.minibatch_count = 0
        self.epoch_count = 0
        self.loss_iters = []
        self.loss_vis = []

        for epoch in range(self.args.last_epoch, self.args.epochs):
            self.update_learning_rate(self.schedulersD, 'netD')
            self.update_learning_rate(self.schedulersG, 'netG')
            self.update_learning_rate(self.schedulersI, 'netI')

            for data in self.train_dataloader:
                # self.idt_img = data['imgA']
                # self.idt_label = data['labelA']
                # self.att_img = data['imgB']
                # self.att_label = data['labelB']
                img, label = data
                self.idt_img = img[0].unsqueeze(0)
                self.idt_label = label[0].unsqueeze(0)
                self.att_img = img[1].unsqueeze(0)
                self.att_label = label[1].unsqueeze(0)

                self.is_same = self.idt_label == self.att_label
                if self.is_same:
                    self.lambdaDiff_alpha = 1.0
                else:
                    self.lambdaDiff_alpha = self.args.lambdaDiff_alpha

                self.batch_size = self.idt_img.size()[0]
                self.epoch_count += 1
                self.minibatch_count += 1
                self.sampler_count += self.batch_size
                if self.use_gpu:
                    self.idt_img = self.idt_img.cuda()
                    self.idt_label = self.idt_label.cuda()
                    self.att_img = self.att_img.cuda()
                    self.att_label = self.idt_label.cuda()

                self.backward_I()
                self.backward_D()
                self.backward_G_A()

                self.loss_current = [
                    self.loss_I.item(),
                    self.loss_A.item(),
                    self.loss_GR.item(),
                    self.loss_C.item(),
                    self.loss_GC.item(),
                    self.loss_GD.item(),
                    self.loss_GAN.item(),
                    self.loss_D_fake.item(),
                    self.loss_D_real.item()
                ]

                self.loss_iters.append(self.loss_current)

                if self.minibatch_count % self.args.display_fre == 0 or self.sampler_count == self.train_data_size:
                    self.visual_results()

    def backward_D(self):
        utils.nn_utils.set_requires_grad(self.netD, True)
        self.optD.zero_grad()
        self.mu, self.logvar, self.feat_A = self.netA(self.att_img)
        self.result = self.netG(self.feat_I, self.feat_A)
        _, self.pred_real = self.netD(self.att_img)
        self.loss_D_real = self.cri_GAN(self.pred_real, True, use_cuda=self.use_gpu)
        _, self.pred_fake = self.netD(self.result)
        self.loss_D_fake = self.cri_GAN(self.pred_fake, False, use_cuda=self.use_gpu)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.args.lambdaGAN
        self.loss_D.backward(retain_graph=True)
        self.optD.step()

    def backward_G_A(self):
        utils.nn_utils.set_requires_grad([self.netD, self.netI], False)
        self.optG.zero_grad()
        self.loss_A = self.cri_Att(self.mu, self.logvar.detach()) * self.args.lambdaA

        self.loss_GR = (self.cri_GR(self.result, self.att_img.detach()) * self.args.lambdaGR * self.lambdaDiff_alpha).mean()

        # self.loss_GR = (self.cri_GR(self.result, self.att_img.detach()) * \
        #                 # self.loss_GR = (self.cri_GR(self.result, self.att_img.detach()).sum(dim=3).sum(dim=2).sum(dim=1) * \
        #                 (self.args.lambdaDiff_alpha * (self.idt_label != self.att_label) + 1. * (
        #                         self.idt_label == self.att_label)).float()).sum() * self.args.lambdaGR

        self.feat_C, self.pred_C = self.netI(self.result)
        self.loss_GC = self.cri_GC(self.feat_C, self.feat_I) * self.args.lambdaGC
        self.loss_C = self.cri_Idt(self.pred_C.squeeze(3).squeeze(2), self.idt_label) * self.args.lambdaCls
        self.feat_D, self.pred_D = self.netD(self.result)
        self.feat_D_, _ = self.netD(self.att_img)
        self.loss_GAN = self.cri_GAN(self.pred_D, True, use_cuda=self.use_gpu) * self.args.lambdaGAN
        self.loss_GD = self.cri_GD(self.feat_D, self.feat_D_.detach()) * self.args.lambdaGD
        self.loss_G = self.loss_A + self.loss_GR + self.loss_C + self.loss_GC + self.loss_GAN + self.loss_GD
        self.loss_G.backward()
        self.optG.step()

    def backward_I(self):
        utils.nn_utils.set_requires_grad(self.netI, True)
        self.optI.zero_grad()
        self.feat_I, self.pred_I = self.netI(self.idt_img)
        self.loss_I = self.cri_Idt(self.pred_I.squeeze(3).squeeze(2), self.idt_label) * self.args.lambdaIdt
        self.loss_I.backward(retain_graph=True)
        self.optI.step()

    def load_model(self):
        if self.args.last_epoch > 0:
            state = torch.load(os.path.join(self.save_dir, str(self.args.last_epoch).zfill(4) + '_state.pth'))
            if self.use_gpu:
                self.netI.module.load_state_dict(state['netI'])
                self.netA.module.load_state_dict(state['netA'])
                self.netG.module.load_state_dict(state['netG'])
                self.netD.module.load_state_dict(state['netD'])
            else:
                self.netI.load_state_dict(state['netI'])
                self.netA.load_state_dict(state['netA'])
                self.netG.load_state_dict(state['netG'])
                self.netD.load_state_dict(state['netD'])
            self.optI.load_state_dict(state['optI'])
            self.optG.load_state_dict(state['optG'])
            self.optD.load_state_dict(state['optD'])
            print('Load model of epoch {:4d}'.format(self.args.last_epoch))
        else:
            utils.nn_utils.init_weights(self.netI, 'I', init_type=self.args.init_type, std=self.args.init_std)
            utils.nn_utils.init_weights(self.netA, 'A', init_type=self.args.init_type, std=self.args.init_std)
            utils.nn_utils.init_weights(self.netG, 'G', init_type=self.args.init_type, std=self.args.init_std)
            utils.nn_utils.init_weights(self.netD, 'D', init_type=self.args.init_type, std=self.args.init_std)

    def save_model(self):
        state = {
            'epoch': self.epoch_count,
            'netI': self.netI.module.state_dict() if self.use_gpu else self.netI.state_dict(),
            'netA': self.netA.module.state_dict() if self.use_gpu else self.netA.state_dict(),
            'netG': self.netG.module.state_dict() if self.use_gpu else self.netG.state_dict(),
            'netD': self.netD.module.state_dict() if self.use_gpu else self.netD.state_dict(),
            'optI': self.optI.state_dict(),
            'optG': self.optG.state_dict(),
            'optD': self.optD.state_dict()
        }
        if (self.epoch_count) % self.args.save_fre == 0:
            torch.save(state, os.path.join(self.args.save_dir, str(self.epoch_count).zfill(4) + '_state.pth'))

        torch.save(state, os.path.join(self.args.save_dir, 'latest_state.pth'))
        with open(os.path.join(self.args.save_dir, 'loss_iters.pkl'), 'wb') as f:
            pickle.dump(self.loss_iters, f)

        save_image(self.vis_imgs,
                   os.path.join(self.save_img_dir, 'epoch_{:s}.jpg'.format(str(self.epoch_count).zfill(4))),
                   nrow=self.args.save_img_cols)

    def visual_results(self):
        self.vis_imgs = [
            self.idt_img[0, :, :, :].cpu().detach(),
            self.att_img[0, :, :, :].cpu().detach(),
            self.result[0, :, :, :].cpu().detach()
        ]
        self.vis_imgs = torch.stack(self.vis_imgs)
        self.vis_imgs = (self.vis_imgs + 1.0) / 2.0
        self.vis.images(self.vis_imgs, nrow=3, win=1)
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
    model = OpensetFaceSwapGANModel(args)
    model.train()
