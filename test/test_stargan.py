# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

import models
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test_stargan(use_gpu=True):
    transform_list = []
    transform_list.append(transforms.Resize(128))
    transform_list.append(transforms.CenterCrop(128))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    transform = transforms.Compose(transform_list)
    test_dataset = utils.datasets.CelebA('./datasets/celeba/Img/img_align_celeba',
                                         './datasets/celeba/Anno/list_attr_celeba.txt',
                                         ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'],
                                         transform, 'test')
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )

    model_choice = 'stargan'
    result_suffix = 'five_attr'
    model = models.stargan.Generator(64, 5, 6)
    model_path = './checkpoints/stargan/celeba/0150_state.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu')['netG'])
    if use_gpu:
        model = model.cuda()
    dst_dir = './results/' + model_choice + '/' + result_suffix
    utils.io.mkdirs(dst_dir)
    if use_gpu:
        model = model.cuda()
    with torch.no_grad():
        index = 0
        for data in test_dataloader:
            real = data[0]
            if use_gpu:
                real = real.cuda()
            fakes = [real.squeeze()]
            for i in range(5):
                label = torch.zeros(5).unsqueeze(0)
                label[0][i] = 1
                if use_gpu:
                    label = label.cuda()
                result = model(real, label).squeeze()
                fakes.append(result)
            final = torch.cat(fakes, dim=2).cpu().numpy()
            final = (final * 0.5 + 0.5) * 255
            final = (np.transpose(final, (1, 2, 0))).astype(np.uint8)
            final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
            index += 1
            cv2.imwrite(os.path.join(dst_dir, str(index).zfill(4) + '.jpg'), final)


if __name__ == '__main__':
    test_stargan(True)
