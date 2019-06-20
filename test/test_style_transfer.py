# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm

import models
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def resize_keep_ratio(img, load_width, load_height, resize_img=cv2.INTER_LINEAR):
    if len(img.shape) == 3:
        height, width, depth = img.shape
    elif len(img.shape) == 2:
        height, width = img.shape
    else:
        raise NotImplementedError('Overall dimension {} is not right'.format(len(img.shape)))
    scale_ratio = max(float(load_height) / height, float(load_width) / width)
    new_img = cv2.resize(img, (0, 0), fx=scale_ratio, fy=scale_ratio, interpolation=resize_img)
    return new_img


def test_pix2pix(use_gpu=True, concat=True):
    model_choice = 'pixpix'
    # result_suffix = 'cityscapes_val'
    result_suffix = 'night2day_val'
    model = models.pix2pix_cyclegan.UnetGenerator(3, 3, 8, 64, norm_layer=nn.BatchNorm2d, use_dropout=False)
    # model_path = './checkpoints/pix2pix/cityscapes256_lambdaI100_batch4/0200_state.pth'
    model_path = './checkpoints/pix2pix/night2day256_lambdaI100_batch4/0200_state.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu')['netG'])
    if use_gpu:
        model = model.cuda()
    # src_dir = './datasets/cityscapes/val'
    src_dir = './datasets/night2day/val'
    dst_dir = './results/' + model_choice + '/' + result_suffix
    utils.io.mkdirs(dst_dir)
    file_num, file_names, file_paths = utils.io.walk_all_files_with_suffix(src_dir)
    if use_gpu:
        model = model.cuda()
    with torch.no_grad():
        for i in tqdm(range(file_num)):
            ori_img = cv2.imread(file_paths[i], -1)
            img = ori_img[:, ori_img.shape[1] // 2: ori_img.shape[1], :]
            dst_img = ori_img[:, 0:ori_img.shape[1] // 2, :]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)
            img = resize_keep_ratio(img, 256, 256)
            dst_img = resize_keep_ratio(dst_img, 256, 256)
            img_tensor = torch.Tensor((np.transpose(img, (2, 0, 1)).astype(np.float) / 255 - 0.5) / 0.5).unsqueeze(0)
            if use_gpu:
                img_tensor = img_tensor.cuda()
            result = model(img_tensor).cpu().squeeze().cpu().numpy()
            result = ((result * 0.5 + 0.5) * 255).astype(np.uint8)
            result = np.transpose(result, (1, 2, 0))
            if concat:
                result = np.concatenate((img, result, dst_img), axis=1)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dst_dir, file_names[i]), result)


def test_cyclegan(use_gpu=True, concat=True):
    model_choice = 'cyclegan'
    # result_suffix = 'orange2apple'
    result_suffix = 'zebra2horse'
    model = models.pix2pix_cyclegan.ResnetGenerator(3, 3, 64, norm_layer=nn.InstanceNorm2d, use_dropout=False,
                                                    n_blocks=9)
    # model_path = './checkpoints/cyclegan/apple2orange256_lambdaI0/0200_state.pth'
    model_path = './checkpoints/cyclegan/horse2zebra256_lambdaI0/0200_state.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu')['netG_B'])
    if use_gpu:
        model = model.cuda()
    # src_dir = './datasets/apple2orange/trainB'
    src_dir = './datasets/horse2zebra/trainB'
    dst_dir = './results/' + model_choice + '/' + result_suffix
    utils.io.mkdirs(dst_dir)
    file_num, file_names, file_paths = utils.io.walk_all_files_with_suffix(src_dir)
    if use_gpu:
        model = model.cuda()
    with torch.no_grad():
        for i in tqdm(range(file_num)):
            img = cv2.imread(file_paths[i], -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = resize_keep_ratio(img, 256, 256)
            img_tensor = torch.Tensor((np.transpose(img, (2, 0, 1)).astype(np.float) / 255 - 0.5) / 0.5).unsqueeze(0)
            if use_gpu:
                img_tensor = img_tensor.cuda()
            result = model(img_tensor).cpu().squeeze().cpu().numpy()
            result = ((result * 0.5 + 0.5) * 255).astype(np.uint8)
            result = np.transpose(result, (1, 2, 0))
            if concat:
                result = np.concatenate((img, result), axis=1)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dst_dir, file_names[i]), result)


def test_att_cyclegan(use_gpu=True, concat=True):
    model_choice = 'att-cyclegan'
    # result_suffix = 'horse2zebra_train_epoch100'
    result_suffix = 'apple2orange_train_epoch100'
    model = models.attention_cyclegan.ResnetGenerator(3, 3, 64, norm_layer=nn.InstanceNorm2d, use_dropout=False,
                                                      n_blocks=9)
    att_model = models.attention_cyclegan.AttentionNet(3, 1, 3, norm_layer=nn.InstanceNorm2d)
    # model_path = './checkpoints/att-cyclegan/horse2zebra256_maskT0.1/0100_state.pth'
    model_path = './checkpoints/att-cyclegan/apple2orange256_maskT0.1/0100_state.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu')['netG_A'])
    att_model.load_state_dict(torch.load(model_path, map_location='cpu')['netA_A'])
    if use_gpu:
        model = model.cuda()
        att_model = att_model.cuda()
    # src_dir = './datasets/horse2zebra/trainA'
    src_dir = './datasets/apple2orange/trainA'
    dst_dir = './results/' + model_choice + '/' + result_suffix
    utils.io.mkdirs(dst_dir)
    file_num, file_names, file_paths = utils.io.walk_all_files_with_suffix(src_dir)
    if use_gpu:
        model = model.cuda()
    with torch.no_grad():
        for i in tqdm(range(file_num)):
            img = cv2.imread(file_paths[i], -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = resize_keep_ratio(img, 256, 256)
            img_tensor = torch.Tensor((np.transpose(img, (2, 0, 1)).astype(np.float) / 255 - 0.5) / 0.5).unsqueeze(0)
            if use_gpu:
                img_tensor = img_tensor.cuda()
            result = model(img_tensor).cpu().squeeze().cpu().numpy()
            result = ((result * 0.5 + 0.5) * 255).astype(np.uint8)
            result = np.transpose(result, (1, 2, 0))
            mask = att_model(img_tensor).cpu().numpy().squeeze()
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            result = (img * (1. - mask) + result * mask).astype(np.uint8)
            mask = (mask * 255).astype(np.uint8)
            if concat:
                result = np.concatenate((img, mask, result), axis=1)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(dst_dir, file_names[i]), result)


if __name__ == '__main__':
    test_pix2pix()
    # test_cyclegan()
    # test_att_cyclegan()
