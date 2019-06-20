# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F
import os
import pickle
from torchvision import transforms
from torchvision.utils import save_image
import sys

sys.path.append('.')

import config
from train import base_model
import models
import utils


class FaderNetModel(base_model.GanModel):

    def get_dataloader(self):
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
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
        transform_list.append(transforms.Normalize(mean, std))
        transform = transforms.Compose(transform_list)
        train_dataset = utils.datasets.CelebA(self.args.train_dir, self.args.celeba_attr_path,
                                              self.args.selected_attrs, transform, 'train')

        train_data_size = len(train_dataset)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.train_batch,
            shuffle=True,
            num_workers=self.args.workers
        )
        return train_data_size, train_loader

    def init_model(self):
        self.args.n_attr = sum([n_cat for _, n_cat in self.args.attr])
        self.netG = models.fadernet.AutoEncoder(self.args)
        self.netD_Latent = models.fadernet.LatentDiscriminator(self.args)
        self.netD_Patch = models.fadernet.PatchDiscriminator(self.args)
        self.netCls = models.fadernet.Classifier(self.args)

        self.n_total_iter = self.args.epochs * self.train_data_size // self.args.train_batch

        utils.nn_utils.print_net(self.netG, 'G')
        utils.nn_utils.print_net(self.netD_Latent, 'D_Latent')
        utils.nn_utils.print_net(self.netD_Patch, 'D_Patch')
        utils.nn_utils.print_net(self.netCls, 'Cls')

        if self.use_gpu:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            self.netG = nn.DataParallel(self.netG.cuda())
            self.netD_Latent = nn.DataParallel(self.netD_Latent.cuda())
            self.netD_Patch = nn.DataParallel(self.netD_Patch.cuda())
            self.netCls = nn.DataParallel(self.netCls.cuda())
        self.optG = torch.optim.Adam(self.netG.parameters(),
                                     lr=self.args.g_lr, betas=(self.args.g_beta1, self.args.g_beta2),
                                     weight_decay=self.args.g_weightdecay)
        self.schedulersG = [utils.nn_utils.get_scheduler(opt, self.args.g_lr_policy, self.args.g_lr_decay_step) for opt
                            in [self.optG]]
        if self.args.n_lat_dis:
            self.optD_Latent = torch.optim.Adam(self.netD_Latent.parameters(),
                                                lr=self.args.d_lr, betas=(self.args.d_beta1, self.args.d_beta2),
                                                weight_decay=self.args.d_weightdecay)
            self.schedulersD_Latent = [
                utils.nn_utils.get_scheduler(opt, self.args.d_lr_policy, self.args.d_lr_decay_step) for opt
                in [self.optD_Latent]]
        if self.args.n_ptc_dis:
            self.optD_Patch = torch.optim.Adam(self.netD_Patch.parameters(),
                                               lr=self.args.d_lr, betas=(self.args.d_beta1, self.args.d_beta2),
                                               weight_decay=self.args.d_weightdecay)
            self.schedulersD_Patch = [
                utils.nn_utils.get_scheduler(opt, self.args.d_lr_policy, self.args.d_lr_decay_step)
                for opt in [self.optD_Patch]]
        if self.args.n_clf_dis:
            self.optCls = torch.optim.Adam(self.netCls.parameters(),
                                           lr=self.args.d_lr, betas=(self.args.d_beta1, self.args.d_beta2),
                                           weight_decay=self.args.d_weightdecay)
            self.schedulersCls = [
                utils.nn_utils.get_scheduler(opt, self.args.d_lr_policy, self.args.d_lr_decay_step)
                for opt in [self.optCls]]

        if self.args.loss_choice == 'vanilla':
            self.cri_GAN = utils.losses.GANLoss(loss_choice='vanilla')
        else:  # lsgan
            self.cri_GAN = utils.losses.GANLoss(loss_choice='lsgan')

    def train(self):
        print('Training Start')
        for net in [self.netD_Latent, self.netG]:
            net.train()
        self.schedulers = [self.schedulersG, self.schedulersD_Latent]

        self.loss_names = ['D_Latent']
        if self.args.n_ptc_dis:
            self.netD_Patch.train()
            self.schedulers.append(self.schedulersD_Patch)
            self.loss_names.append('D_Patch')
        if self.args.n_clf_dis:
            self.netCls.train()
            self.schedulers.append(self.schedulersCls)
            self.loss_names.append('Cls')
        self.loss_names.append('G')

        self.sampler_count = 0
        self.minibatch_count = 0
        self.epoch_count = 0
        self.loss_iters = []
        self.loss_vis = []

        for epoch in range(self.args.last_epoch, self.args.epochs):
            self.update_learning_rate(self.schedulersG, 'netG')
            self.update_learning_rate(self.schedulersD_Latent, 'netD_Latent')
            if self.args.n_ptc_dis:
                self.update_learning_rate(self.schedulersD_Patch, 'netD_Patch')
            if self.args.n_clf_dis:
                self.update_learning_rate(self.schedulersCls, 'netCls')
            self.epoch_count += 1
            for data in self.train_dataloader:
                self.data = data
                self.loss_current = []
                for _ in range(self.args.n_lat_dis):
                    self.backward_D_Latent()
                for _ in range(self.args.n_ptc_dis):
                    self.backward_D_Patch()
                for _ in range(self.args.n_clf_dis):
                    self.backward_Cls()
                self.backward_G()

                self.batch_size = self.data[0].size(0)
                self.minibatch_count += 1
                self.sampler_count += self.batch_size
                self.loss_iters.append(self.loss_current)
                if self.minibatch_count % self.args.display_fre == 0 or self.sampler_count == self.train_data_size:
                    self.visual_results()
                    self.save_model()

    def backward_D_Latent(self):
        data = self.data
        params = self.args
        self.netG.eval()
        self.netD_Latent.train()
        # batch / encode / discriminate
        batch_x, label_y = data
        if self.use_gpu:
            batch_x = batch_x.cuda()
            label_y = label_y.cuda()
            enc_outputs = self.netG.module.encode(batch_x)
        else:
            enc_outputs = self.netG.encode(batch_x)
        batch_y = torch.zeros(label_y.size(0), params.attr[0][1])
        batch_y.scatter_(1, label_y.long(), 1)
        preds = self.netD_Latent(enc_outputs[-1 - params.n_skip])
        loss = models.fadernet.get_attr_loss(preds, batch_y, False, params)
        # loss / optimize
        self.optD_Latent.zero_grad()
        if len(self.loss_current) == 0:
            self.loss_current.append(loss.item())
        loss.backward()
        if params.clip_grad_norm:
            utils.nn_utils.clip_grad_norm(self.netD_Latent.parameters(), params.clip_grad_norm)
        self.optD_Latent.step()

    def backward_D_Patch(self):
        data = self.data
        params = self.args
        self.netG.eval()
        self.netD_Patch.train()
        # batch / encode / discriminate
        batch_x, label_y = data
        if self.use_gpu:
            batch_x = batch_x.cuda()
            label_y = label_y.cuda()
        batch_y = torch.zeros(label_y.size(0), params.attr[0][1])
        batch_y.scatter_(1, label_y.long(), 1)
        flipped = models.fadernet.flip_attributes(batch_y, params, 'all')
        _, dec_outputs = self.netG((batch_x), flipped)
        real_preds = self.netD_Patch(batch_x)
        fake_preds = self.netD_Patch(dec_outputs[-1].data)
        y_fake = torch.FloatTensor(real_preds.size()).fill_(params.smooth_label).cuda()
        # loss / optimize
        loss = F.binary_cross_entropy(real_preds, 1 - y_fake)
        loss += F.binary_cross_entropy(fake_preds, y_fake)
        self.optD_Patch.zero_grad()
        if len(self.loss_current) == 1:
            self.loss_current.append(loss.item())
        loss.backward()
        if params.clip_grad_norm:
            utils.nn_utils.clip_grad_norm(self.netD_Patch.parameters(), params.clip_grad_norm)
        self.optD_Patch.step()

    def backward_Cls(self):
        data = self.data
        params = self.args
        self.netCls.train()
        # batch / predict
        batch_x, label_y = data
        if self.use_gpu:
            batch_x = batch_x.cuda()
            label_y = label_y.cuda()
        batch_y = torch.zeros(label_y.size(0), params.attr[0][1])
        batch_y.scatter_(1, label_y.long(), 1)
        preds = self.netCls(batch_x)
        # loss / optimize
        loss = models.fadernet.get_attr_loss(preds, batch_y, False, params)
        self.optCls.zero_grad()
        if len(self.loss_current) == 2:
            self.loss_current.append(loss.item())
        loss.backward()
        if params.clip_grad_norm:
            utils.nn_utils.clip_grad_norm(self.netCls.parameters(), params.clip_grad_norm)
        self.optCls.step()

    def backward_G(self):
        data = self.data
        params = self.args
        self.netG.train()
        if params.n_lat_dis:
            self.netD_Latent.eval()
        if params.n_ptc_dis:
            self.netD_Patch.eval()
        if params.n_clf_dis:
            self.netCls.eval()
        # batch / encode / decode
        batch_x, label_y = data
        if self.use_gpu:
            batch_x = batch_x.cuda()
            label_y = label_y.cuda()
        batch_y = torch.zeros(label_y.size(0), params.attr[0][1])
        batch_y.scatter_(1, label_y.long(), 1)

        enc_outputs, dec_outputs = self.netG(batch_x, batch_y)
        # autoencoder loss from reconstruction
        loss = params.lambda_ae * ((batch_x - dec_outputs[-1]) ** 2).mean()
        self.real = batch_x
        self.fake = dec_outputs[-1]
        # encoder loss from the latent discriminator
        if params.lambda_lat_dis:
            lat_dis_preds = self.netD_Latent(enc_outputs[-1 - params.n_skip])
            lat_dis_loss = models.fadernet.get_attr_loss(lat_dis_preds, batch_y, True, params)
            loss = loss + models.fadernet.get_lambda(params.lambda_lat_dis, self.n_total_iter,
                                                     self.args.lambda_schedule) * lat_dis_loss
        # decoding with random labels
        if params.lambda_ptc_dis + params.lambda_clf_dis > 0:
            flipped = models.fadernet.flip_attributes(batch_y, params, 'all')
            dec_outputs_flipped = self.ae.decode(enc_outputs, flipped)
        # autoencoder loss from the patch discriminator
        if params.lambda_ptc_dis:
            ptc_dis_preds = self.netD_Patch(dec_outputs_flipped[-1])
            y_fake = torch.FloatTensor(ptc_dis_preds.size()).fill_(params.smooth_label).cuda()
            ptc_dis_loss = F.binary_cross_entropy(ptc_dis_preds, 1 - y_fake)
            loss = loss + models.fadernet.get_lambda(params.lambda_ptc_dis, self.n_total_iter,
                                                     self.args.lambda_schedule) * ptc_dis_loss
        # autoencoder loss from the classifier discriminator
        if params.lambda_clf_dis:
            clf_dis_preds = self.netCls(dec_outputs_flipped[-1])
            clf_dis_loss = models.fadernet.get_attr_loss(clf_dis_preds, flipped, False, params)
            loss = loss + models.fadernet.get_lambda(params.lambda_clf_dis, self.n_total_iter,
                                                     self.args.lambda_schedules) * clf_dis_loss
        # check NaN
        # optimize
        self.optG.zero_grad()
        self.loss_current.append(loss.item())
        loss.backward()
        if params.clip_grad_norm:
            utils.nn_utils.clip_grad_norm(self.netG.parameters(), params.clip_grad_norm)
        self.optG.step()

    def load_model(self):
        if self.args.last_epoch > 0:
            state = torch.load(os.path.join(self.save_dir, str(self.args.last_epoch).zfill(4) + '_state.pth'))
            if self.use_gpu:
                self.netG.module.load_state_dict(state['netG'])
                self.netD_Latent.module.load_state_dict(state['netD_Latent'])
                self.netD_Patch.module.load_state_dict(state['netD_Patch'])
                self.netCls.module.load_state_dict(state['netCls'])
            else:
                self.netG.load_state_dict(state['netG'])
                self.netD_Latent.load_state_dict(state['netD_Latent'])
                self.netD_Patch.load_state_dict(state['netD_Patch'])
                self.netCls.load_state_dict(state['netCls'])
            print('Load model of epoch {:4d}'.format(self.args.last_epoch))
        else:
            pass
            # official code uses default init of pytorch
            # utils.nn_utils.init_weights(self.netG, 'G', init_type=self.args.init_type, std=self.args.init_std)
            # utils.nn_utils.init_weights(self.netD_Latent, 'D_Latent', init_type=self.args.init_type,
            #                             std=self.args.init_std)
            # utils.nn_utils.init_weights(self.netD_Patch, 'D_Patch', init_type=self.args.init_type,
            #                             std=self.args.init_std)
            # utils.nn_utils.init_weights(self.netCls, 'Cls', init_type=self.args.init_type, std=self.args.init_std)

    def save_model(self):
        # celeba is too large, save by iterations is better
        state = {
            'epoch': self.epoch_count,
            'netG': self.netG.module.state_dict() if self.use_gpu else self.netG.state_dict(),
            'netD_Latent': self.netD_Latent.module.state_dict() if self.use_gpu else self.netD_Latent.state_dict(),
            'netD_Patch': self.netD_Patch.module.state_dict() if self.use_gpu else self.netD_Patch.state_dict(),
            'netCls': self.netCls.module.state_dict() if self.use_gpu else self.netD.state_dict(),
        }
        if self.epoch_count % self.args.save_fre == 0:
            torch.save(state, os.path.join(self.args.save_dir, str(self.epoch_count).zfill(4) + '_state.pth'))
            save_imgs = self.vis_imgs
            save_image(save_imgs,
                       os.path.join(self.save_img_dir, 'epoch_{:s}.jpg'.format(str(self.epoch_count).zfill(4))),
                       nrow=self.args.save_img_cols)
        torch.save(state, os.path.join(self.args.save_dir, 'latest_state.pth'))
        with open(os.path.join(self.args.save_dir, 'loss_iters.pkl'), 'wb') as f:
            pickle.dump(self.loss_iters, f)

    def visual_results(self):
        self.vis_imgs = [
            self.real[0:4, :, :, :].cpu().detach(),
            self.fake[0:4, :, :, :].cpu().detach(),
        ]
        self.vis_imgs = torch.cat(self.vis_imgs, dim=0)
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
    model = FaderNetModel(args)
    model.train()
