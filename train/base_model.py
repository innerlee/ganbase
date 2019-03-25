# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision import transforms
from visdom import Visdom
from torch.utils.data import DataLoader
import pickle
from torchvision.utils import save_image
import os
import json

import models
import utils


class GanModel(object):
    def __init__(self, args):
        super(GanModel, self).__init__()

        self.args = args
        # begin prepare
        self.prepare_model()
        # end prepare

        # begin data_loader
        self.train_data_size, self.train_dataloader = self.get_dataloader()
        # end data_loader

        # begin init_model
        self.init_model()
        # end init_model

        # begin load_model
        self.load_model()
        # end load_model

        # begin visual
        self.vis = Visdom(port=self.args.visdom_port, env=self.args.visdom_env)
        # end visual

    def train(self):
        pass

    def backward_D(self):
        pass

    def backward_D_gp(self):
        pass

    def backward_G(self):
        pass

    def update_learning_rate(self, schedulers, name):
        for scheduler in schedulers:
            scheduler.step()
        lr = self.optG.param_groups[0]['lr']
        print('learning rate of {} = {:.7f}'.format(name, lr))

    def optimize_parameters(self):
        pass

    def get_dataloader(self):
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        train_transform = utils.datasets.get_pil_transform('train', self.args.resize_choice, self.args.load_size,
                                                           self.args.fine_size, self.args.flip.choice, mean, std)

        train_dataset = utils.datasets.PureDataset(data_dir=self.args.train_dir, transform=train_transform)
        train_data_size = len(train_dataset)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.train_batch,
            shuffle=True,
            num_workers=self.args.workers
        )
        return train_data_size, train_loader

    def prepare_model(self):
        gpu = self.args.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        self.num_gpu = torch.cuda.device_count()
        self.use_gpu = torch.cuda.is_available() and self.num_gpu > 0
        self.save_img_num = self.args.save_img_rows * self.args.save_img_cols
        self.save_inputs = torch.rand(self.save_img_num, self.args.g_input_dim)
        if self.use_gpu:
            self.save_inputs = self.save_inputs.cuda()
        self.save_img_dir = os.path.join(self.args.save_dir, 'images')
        self.save_dir = self.args.save_dir
        utils.io.mkdirs(self.save_dir)
        utils.io.mkdirs(self.save_img_dir)
        print('Save dir created: {:s}'.format(self.args.save_dir))

        with open(os.path.join(self.args.save_dir, 'args.txt'), 'a') as f:
            f.write('last epoch: {:d} \n'.format(self.args.last_epoch))
            f.write(json.dumps(vars(self.args), indent=4))
            f.write('\n')

    def init_model(self):
        g_norm_layer = utils.nn_utils.get_norm_layer(norm_type=self.args.g_norm_type)
        d_norm_layer = utils.nn_utils.get_norm_layer(norm_type=self.args.d_norm_type)
        self.netG = models.dcgan.Generator(imsize=self.args.load_size, imchannel=self.args.d_input_dim, width=16,
                                           nz=self.args.g_input_dim, normalize=g_norm_layer)
        self.netD = models.dcgan.Discriminator(imsize=self.args.load_size, imchannel=self.args.d_input_dim,
                                               width=16, normalize=d_norm_layer)

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
        self.schedulersG = [utils.nn_utils.get_scheduler(opt, self.args.g_lr_policy, self.args.g_lr_decay_step) for opt
                            in
                            [self.optG]]
        self.schedulersD = [utils.nn_utils.get_scheduler(opt, self.args.d_lr_policy, self.args.d_lr_decay_step) for opt
                            in
                            [self.optD]]

        if self.args.loss_choice == 'vanilla':
            self.cri = utils.losses.GANLoss(loss_choice='gan')  # gan
        elif self.args.loss_choice == 'lsgan':
            self.cri = utils.losses.GANLoss(loss_choice='lsgan')  # lsgan
        else:
            self.cri = utils.losses.GANLoss(loss_choice='gan')

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
            utils.nn_utils.init_weights(self.netG, 'G')
            utils.nn_utils.init_weights(self.netD, 'D')

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

        save_imgs = (self.netG(self.save_inputs).detach().cpu() + 1.0) / 2.0
        save_image(save_imgs, os.path.join(self.save_img_dir, 'epoch_{:s}.jpg'.format(str(self.epoch_count).zfill(4))),
                   nrow=self.args.save_img_cols)

    def visual_results(self):
        save_imgs = (self.netG(self.save_inputs).detach().cpu() + 1.0) / 2.0
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
    import sys

    sys.path.append('.')
    import config

    args = config.train_config()
    model = GanModel(args)
