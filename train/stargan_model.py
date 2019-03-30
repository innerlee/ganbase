# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import pickle
from torchvision.utils import save_image
from torch.autograd import grad
import torch.nn.functional as F
from torch.backends import cudnn

from train import base_model
import models
import utils


class StarGANModel(base_model.GanModel):

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

        train_dataset = utils.datasets.CelebA(self.args.train_dir, self.args.celeba_attr_path, self.args.selected_attrs,
                                              transform, 'train')

        train_data_size = len(train_dataset)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.train_batch,
            shuffle=True,
            num_workers=self.args.workers
        )
        return train_data_size, train_loader

    def init_model(self):

        self.netG = models.stargan.Generator(self.args.g_net_width, self.args.attr_dim, self.args.g_resnet_blocks)

        self.netD = models.stargan.Discriminator(self.args.fine_size, self.args.d_net_width,
                                                 self.args.attr_dim, self.args.d_resnet_blocks)

        utils.nn_utils.print_net(self.netG, 'G')
        utils.nn_utils.print_net(self.netD, 'D')

        if self.use_gpu:
            self.netG = nn.DataParallel(self.netG.cuda())
            self.netD = nn.DataParallel(self.netD.cuda())
            cudnn.benchmark = True

        self.optG = torch.optim.Adam(self.netG.parameters(),
                                     lr=self.args.g_lr, betas=(self.args.g_beta1, self.args.g_beta2),
                                     weight_decay=self.args.g_weightdecay)
        self.optD = torch.optim.Adam(self.netD.parameters(),
                                     lr=self.args.d_lr, betas=(self.args.d_beta1, self.args.d_beta2),
                                     weight_decay=self.args.d_weightdecay)
        self.schedulersG = [utils.nn_utils.get_scheduler(opt, self.args.g_lr_policy, self.args.g_lr_decay_step,
                                                         lr_gamma=self.args.g_lr_decay_gamma) for opt
                            in [self.optG]]
        self.schedulersD = [utils.nn_utils.get_scheduler(opt, self.args.d_lr_policy, self.args.d_lr_decay_step,
                                                         lr_gamma=self.args.d_lr_decay_gamma) for opt
                            in [self.optD]]
        self.cri_Rec = nn.L1Loss()

    def train(self):
        print('Training Start')
        for net in [self.netD, self.netG]:
            net.train()
        self.schedulers = [self.schedulersD, self.schedulersG]
        self.loss_names = ['D_real', 'D_fake', 'D_gp', 'D_cls', 'G_GAN', 'G_rec', 'G_cls']
        self.sampler_count = 0
        self.minibatch_count = 0
        self.epoch_count = 0
        self.loss_iters = []
        self.loss_vis = []

        self.update_learning_rate(self.schedulersD, 'netD')
        self.update_learning_rate(self.schedulersG, 'netG')

        for epoch in range(self.args.last_epoch, self.args.epochs):
            self.epoch_count += 1

            for data in self.train_dataloader:
                self.real = data[0]
                self.label = data[1]
                self.batch_size = self.real.size()[0]
                self.minibatch_count += 1
                self.sampler_count += self.batch_size
                if self.use_gpu:
                    self.real = self.real.cuda()
                    self.label = self.label.cuda()

                rand_idx = torch.randperm(self.label.size(0))
                self.label_trg = self.label[rand_idx]

                self.backward_D()
                if self.minibatch_count % 5 == 0:
                    self.backward_G()
                    self.loss_current = [
                        self.loss_D_real,
                        self.loss_D_fake,
                        self.loss_D_gp,
                        self.loss_D_cls,
                        self.loss_G_GAN,
                        self.loss_G_rec,
                        self.loss_G_cls
                    ]
                    self.loss_iters.append(self.loss_current)

                if self.minibatch_count % self.args.display_fre == 0 or self.sampler_count == self.train_data_size:
                    self.visual_results()
            self.save_model()

    def backward_D(self):
        self.optD.zero_grad()
        D_real, real_cls = self.netD(self.real)
        self.fake = self.netG(self.real, self.label_trg)
        D_fake, fake_cls = self.netD(self.fake)
        loss_D_real = -torch.mean(D_real)
        loss_D_fake = torch.mean(D_fake)
        loss_D_cls = F.binary_cross_entropy_with_logits(real_cls, self.label, reduction='sum') / real_cls.size(0)

        alpha = torch.rand(self.batch_size, 1, 1, 1)
        if self.use_gpu:
            alpha = alpha.cuda()
        x_hat = alpha * self.real.data + (1.0 - alpha) * self.fake.data
        x_hat.requires_grad = True
        pred_hat, _ = self.netD(x_hat)
        if self.use_gpu:
            gradients = \
                grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                     create_graph=True, retain_graph=True)[0]
        else:
            gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                             create_graph=True, retain_graph=True)[0]
        loss_D_gp = ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
        loss_D = loss_D_real + loss_D_fake + self.args.lambdaCls * loss_D_cls + self.args.lambdaGP * loss_D_gp
        loss_D.backward()

        self.loss_D_real = loss_D_real.item()
        self.loss_D_fake = loss_D_fake.item()
        self.loss_D_cls = loss_D_cls.item()
        self.loss_D_gp = loss_D_gp.item()

        self.optD.step()

    def backward_G(self):
        self.optG.zero_grad()

        fake = self.netG(self.real, self.label_trg)
        D_fake, fake_cls = self.netD(fake)
        self.rec = self.netG(fake, self.label)
        loss_G_GAN = - torch.mean(D_fake)
        loss_G_rec = self.cri_Rec(self.rec, self.real)
        loss_G_cls = F.binary_cross_entropy_with_logits(fake_cls, self.label_trg, reduction='sum') / fake_cls.size(0)

        loss_G = loss_G_GAN + self.args.lambdaCls * loss_G_cls + self.args.lambdaRec * loss_G_rec

        loss_G.backward()
        self.loss_G_GAN = loss_G_GAN.item()
        self.loss_G_cls = loss_G_cls.item()
        self.loss_G_rec = loss_G_rec.item()
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
            # utils.nn_utils.init_weights(self.netG, 'G', self.args.init_type, self.args.init_std)
            # utils.nn_utils.init_weights(self.netD, 'D', self.args.init_type, self.args.init_std)
            # Official code has not init func
            pass

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

        # save_imgs = self.vis_imgs
        # save_image(save_imgs, os.path.join(self.save_img_dir, 'epoch_{:s}.jpg'.format(str(self.epoch_count).zfill(4))),
        #            nrow=self.args.save_img_cols)

    def visual_results(self):
        self.vis_imgs = [
            self.real[0, :, :, :].cpu().detach(),
            self.fake[0, :, :, :].cpu().detach(),
            self.rec[0, :, :, :].cpu().detach()
        ]
        self.vis_imgs = torch.stack(self.vis_imgs).numpy()
        self.vis_imgs = (self.vis_imgs + 1.0) / 2.0
        self.vis.images(self.vis_imgs * 255, nrow=4, win=1)
        self.loss_vis.append(self.loss_current)
        utils.visual.show_visdom_line(self.vis, self.loss_vis, self.loss_names,
                                      win=2, xlabel='{:d} iters'.format(self.args.display_fre))
        s = 'epoch: {:4d} iters: {:8d}  '.format(
            self.epoch_count, self.minibatch_count)
        for i in range(len(self.loss_names)):
            s += self.loss_names[i] + ': {:.6f}  '.format(self.loss_current[i])
        print(s)


if __name__ == '__main__':
    import sys

    sys.path.append('.')
    import config

    args = config.train_config()
    model = StarGANModel(args)
    model.train()
