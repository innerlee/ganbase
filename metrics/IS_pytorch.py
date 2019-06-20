# -*- coding: utf-8 -*-

# !/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import glob
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
import numpy as np
# from scipy.misc import imread
import cv2
from torch.nn.functional import adaptive_avg_pool2d

from Inception_v3_pytorch import InceptionV3
import torch.nn as nn
from tqdm import tqdm


def get_activations(images, model, batch_size=100, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                     must lie between 0 and 1.
    -- model       : Instance of inception model
    -- batch_size  : the images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size depends
                     on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    d0 = images.shape[0]
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    # pred_arr = np.empty((n_used_imgs, dims))
    # TODO change default dim to 1000
    pred_arr = np.empty((n_used_imgs, 1000))
    print('Model output...')
    with torch.no_grad():
        for i in tqdm(range(n_batches)):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                      end='', flush=True)
            start = i * batch_size
            end = start + batch_size

            batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
            # batch = Variable(batch, volatile=True)
            if cuda:
                batch = batch.cuda()
            pred = model(batch)[0]
            pred = pred.view(pred.size(0), -1)
            pred = model.fc(pred)
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.dim() == 4:
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_inception_score(images, model, batch_size=100, splits=10,
                              dims=2048, cuda=False, verbose=False, eps=1e-8):
    preds = get_activations(images, model, batch_size, dims, cuda, verbose)
    preds = torch.from_numpy(preds)  # N x C
    preds = nn.Softmax(dim=1)(preds).numpy()
    scores = []
    print('IS result...')
    for i in tqdm(range(splits)):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    # scores = np.concatenate(scores, 0)
    return np.mean(scores), np.std(scores)


def _compute_statistics_of_path(path, model, batch_size, nsamples, splits, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        # path = pathlib.Path(path)
        # files = list(path.glob('*.jpg')) + list(path.glob('*.png'))

        files = glob.glob(path + '/**/*.jpg', recursive=True) \
                + glob.glob(path + '/**/*.png', recursive=True) \
                + glob.glob(path + '/*.jpg') \
                + glob.glob(path + '/*.png')

        new_files = random.sample(files, nsamples)
        # imgs = np.array([imread(str(fn)).astype(np.float32) for fn in new_files])
        imgs = []
        for i in tqdm(range(len(new_files))):
            # imgs.append(imread(str(new_files[i])).astype(np.float32))
            # imgs.append(cv2.imread(new_files[i], -1).astype(np.float32))
            imgs.append(cv2.cvtColor(cv2.imread(new_files[i], -1), cv2.COLOR_BGR2RGB).astype(np.float32))
        imgs = np.array(imgs)
        # Bring images to shape (B, 3, H, W)
        imgs = imgs.transpose((0, 3, 1, 2))
        # Rescale images to be between 0 and 1
        imgs /= 255

        m, s = calculate_inception_score(imgs, model, batch_size, splits, dims, cuda)
    return m, s


def calculate_is_given_path(path, batch_size, nsamples, splits, cuda, dims):
    if not os.path.exists(path):
        raise RuntimeError('Invalid path: %s' % path)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()
    m, s = _compute_statistics_of_path(path, model, batch_size, nsamples, splits, dims, cuda)
    return m, s


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, help='Path to image dir')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size to use')
    parser.add_argument('--splits', type=int, default=10, help='Splits to cal')
    parser.add_argument('--nsamples', type=int, help='The number of randomly selected samples')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('--gpu', default='', type=str, help='GPU to use (leave blank for CPU only)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    is_m, is_s = calculate_is_given_path(args.path,
                                         args.batch_size,
                                         args.nsamples,
                                         args.splits,
                                         args.gpu != '',
                                         args.dims)
    print(is_m, is_s)
