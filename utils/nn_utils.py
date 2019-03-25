# -*- coding: utf-8 -*-

import functools
import torch.nn as nn
import torch.nn.init as init
from torch.optim import lr_scheduler
import torch
import random
from collections import deque
import os


def print_net(net, name):
    count = 0
    for param in net.parameters():
        count += param.numel()
    print('# of params in {:s} is {:10d}'.format(name, count))


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_act_layer(act_type='relu'):
    if act_type == 'relu':
        act_layer = nn.ReLU()
    elif act_type == 'lrelu':
        act_layer = nn.LeakyReLU(0.2)
    elif act_type == 'elu':
        act_layer = nn.ELU()
    else:
        act_layer = None
    return act_layer


def init_weights(net, name, init_type='kaiming_uniform', mean=0, std=1, bound=1, gain=1, a=1, mode='fan_in',
                 nonlinearty='relu'):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, mean=mean, std=std)
            elif init_type == 'uniform':
                init.uniform_(m.weight.data, a=-bound, b=bound)
            elif init_type == 'xavier_norm':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'kaiming_norm':
                init.kaiming_normal_(m.weight.data, a=a, mode=mode, nonlinearity=nonlinearty)
            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=a, mode=mode, nonlinearity=nonlinearty)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)

    print('initialize network {:s} with {:s}'.format(name, init_type))
    net.apply(init_func)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def load_pretrained_dict(old_model, new_model):
    old_state_dict = old_model.state_dict()
    new_state_dict = new_model.state_dict()
    old_count = 0
    new_count = 0
    for k, v in old_state_dict.items():
        old_count += 1
        if k in new_state_dict:
            new_count += 1
            new_state_dict[k] = v
    new_model.load_state_dict(new_state_dict)
    print('{}/{} loaded'.format(new_count, old_count))


def load_pretrained_dict_ordered(old_model, new_model):
    old_state_dict = old_model.state_dict()
    new_state_dict = new_model.state_dict()
    new_count = 0
    old_param_list = list(old_state_dict.values())
    for k, v in new_state_dict.items():
        new_state_dict[k] = old_param_list[new_count]
        new_count += 1
    new_model.load_state_dict(new_state_dict)
    print('{} parts loaded'.format(new_count))


def get_scheduler(optimizer, lr_policy='lambda', lr_decay_step=30, lr_decay_steps=[30000], lr_gamma=0.1,
                  last_epoch=-1, warm_up_steps=100, linear_steps=100):
    def exp_lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 + 1 - 100) / float(100 + 1)
        return lr_l

    def linear_lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - warm_up_steps) / float(linear_steps + 1)
        return lr_l

    if lr_policy == 'lambda':
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=exp_lambda_rule, last_epoch=last_epoch)
    elif lr_policy == 'linear':
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_lambda_rule, last_epoch=last_epoch)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_gamma, last_epoch=last_epoch)
    elif lr_policy == 'multi_steps':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_steps, gamma=lr_gamma,
                                             last_epoch=last_epoch)
    elif lr_policy == 'exp_lr':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma, last_epoch=last_epoch)
    elif lr_policy == 'cos_lr':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000000, eta_min=0, last_epoch=-1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


class ImagePool(object):
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


class Saver(object):
    """
    A wrapper around `torch.save`.

    args
    `slot`: how many models to save for each iteration
    `keepnum`: maximum number of files saved, default 3.
        0 for no save,
        -1 for saving all.
    """

    def __init__(self, slot, keepnum=3):
        self.snap = deque([])
        self.slot = slot
        self.keepnum = keepnum

    def save(self, obj, f, iter):
        state = {'model': obj, 'iter': iter}
        torch.save(state, f)
        self.snap.append(f)
        if len(self.snap) == (self.keepnum + 1) * self.slot:
            for _ in range(self.slot):
                os.remove(self.snap.popleft())

    def load(self, obj, f):
        state = torch.load(f)
        obj.load_state_dict(state['model'])
        iter = state['iter']
        return obj, iter
