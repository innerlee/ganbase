# -*- coding: utf-8 -*-

import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random
from torch.utils import data
import torch
import numpy as np


def walk_all_files_with_suffix(dir, suffixs=('.jpg', '.png')):
    paths = []
    names = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            if os.path.splitext(fname)[1] in suffixs:
                paths.append(path)
                names.append(fname)
    return len(names), names, paths


def get_crop_params(load_size, crop_size):
    x = random.randint(0, np.maximum(0, load_size - crop_size))
    y = random.randint(0, np.maximum(0, load_size - crop_size))
    return x, y


def get_flip_params(prob=0.5):
    flip = random.random() > prob
    return flip


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip, direction=0):
    if flip and direction == 0:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip and direction == 1:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def get_pil_transform(phase='train', resize_choice=2, load_size=286, fine_size=256, flip_choice=1,
                      mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), params=None):
    """

    :param phase: train or test
    :param resize_choice: 0 for load, 1 for resize, 2 for resize and random-crop, 3 for resize and center-crop
    :param load_size: image size when loading image
    :param fine_size: image size for the network input
    :param flip_choice: 0 for not flip, 1 for horizontal flip, 2 for vertical-flip
    :param mean:  mean pixel value for RGB
    :param std: std pixel value for RGB
    :param params: param for RandomCrop and RandomFlip
    :return:
    """

    transform_list = []
    if phase == 'train':
        if resize_choice > 0:
            transform_list.append(transforms.Resize(load_size, interpolation=Image.BICUBIC))
        if resize_choice == 2:
            transform_list.append(transforms.CenterCrop(fine_size))
        elif resize_choice == 3:
            if params is None:
                transform_list.append(transforms.RandomCrop(fine_size))
            else:
                transform_list.append(
                    transforms.Lambda(lambda img: __crop(img, params['crop_pos'], fine_size)))
        else:
            raise NotImplementedError('Resize choice {} is not supported yte'.format(resize_choice))
        if flip_choice == 1:
            if params is None:
                transform_list.append(transforms.RandomHorizontalFlip())
            else:
                transform_list.append(transforms.Lambda(lambda img: __flip(img, __flip(img, params['hor_flip'], 0))))
        elif flip_choice == 2:
            if params is None:
                transform_list.append(transforms.RandomVerticalFlip())
            else:
                transform_list.append(transforms.Lambda(lambda img: __flip(img, __flip(img, params['ver_flip'], 1))))
        else:
            raise NotImplementedError('Flip choice {} is not supported yte'.format(flip_choice))
    elif phase == 'test':
        if resize_choice > 0:
            transform_list.append(transforms.Resize(load_size, interpolation=Image.BICUBIC))
        if resize_choice > 1:
            transform_list.append(transforms.CenterCrop(fine_size))
        else:
            raise NotImplementedError('Resize choice {} is not supported yte'.format(resize_choice))
    else:
        raise NotImplementedError('Phase {} is not supported yet'.format(phase))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize(mean, std)]
    return transforms.Compose(transform_list)


class PureDataset(Dataset):
    def __init__(self, data_dir, transform):
        super(PureDataset, self).__init__()
        self.data_dir = data_dir
        self.file_num, self.file_names, self.file_paths = walk_all_files_with_suffix(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        img = Image.open(file_path).convert('RGB')
        img = self.transform(img)
        return {'img': img, 'path': file_path}

    def __len__(self):
        return self.file_num


class Pix2PixDataset(Dataset):
    def __init__(self, dir, args, mean, std, phase='train'):
        super(Pix2PixDataset, self).__init__()
        self.file_num, self.file_names, self.file_paths = walk_all_files_with_suffix(dir)
        self.args = args
        self.phase = phase
        self.mean = mean
        self.std = std

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        img = Image.open(file_path).convert('RGB')
        w, h = img.size
        img_A = img.crop((0, 0, w // 2, h))
        img_B = img.crop((w // 2, 0, w, h))

        crop_pos = get_crop_params(self.args.load_size, self.args.fine_size)
        hor_flip = get_flip_params()
        params = {'crop_pos': crop_pos, 'hor_flip': hor_flip, 'ver_flip': False}
        transform = get_pil_transform(self.phase, self.args.resize_choice, self.args.load_size, self.args.fine_size,
                                      self.args.flip, self.mean, self.std, params)
        A = transform(img_A)
        B = transform(img_B)
        # inverse order as paper
        return {'A': B, 'B': A, 'path': file_path}

    def __len__(self):
        return self.file_num


class UnalignedDataset2(Dataset):
    def __init__(self, data_dir_A, data_dir_B, transform, serial_batch=False):
        super(UnalignedDataset2, self).__init__()
        self.dir_A = data_dir_A
        self.dir_B = data_dir_B
        self.size_A, self.names_A, self.paths_A = walk_all_files_with_suffix(self.dir_A)
        self.size_B, self.names_B, self.paths_B = walk_all_files_with_suffix(self.dir_B)
        self.serial_batch = serial_batch
        self.transform = transform

    def __getitem__(self, index):
        # 防止因为B更多出现溢出
        index_A = index % self.size_A
        if self.serial_batch:
            index_B = index % self.size_B
        else:
            index_B = random.randint(0, self.size_B - 1)

        path_A = self.paths_A[index_A]
        path_B = self.paths_B[index_B]
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        A = self.transform(img_A)
        B = self.transform(img_B)
        return {'A': A, 'B': B, 'path_A': path_A, 'path_B': path_B}

    def __len__(self):
        return max(self.size_A, self.size_B)


class AlignedDataset2(Dataset):
    def __init__(self, data_dir_A, data_dir_B, transform, serial_batch=False):
        super(AlignedDataset2, self).__init__()
        self.dir_A = data_dir_A
        self.dir_B = data_dir_B
        self.size_A, self.names_A, self.paths_A = walk_all_files_with_suffix(self.dir_A)
        self.size_B, self.names_B, self.paths_B = walk_all_files_with_suffix(self.dir_B)
        self.serial_batch = serial_batch
        self.transform = transform
        if self.size_A != self.size_B:
            print('size of two dirs should be the same')
            raise AssertionError
        for i in range(self.size_A):
            if self.names_A[i] != self.names_B[i]:
                print(self.names_A[i] + ' ' + self.names_B[i])
                raise AssertionError

    def __getitem__(self, index):
        path_A = self.paths_A[index]
        path_B = self.paths_B[index]
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        A = self.transform(img_A)
        B = self.transform(img_B)
        return {'A': A, 'B': B, 'path_A': path_A, 'path_B': path_B}

    def __len__(self):
        return max(self.size_A, self.size_B)


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i + 1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images
