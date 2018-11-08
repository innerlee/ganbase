
import os
import pathlib

import torch
import numpy as np
from scipy.misc import imread
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d

from ganbase.evaluation.inception import InceptionV3
import torch.nn as nn


def get_activations(images, model, batch_size=100, dims=2048,
                    cuda=False, verbose=False):
    model.eval()

    d0 = images.shape[0]
    if batch_size > d0:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = d0

    n_batches = d0 // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))
    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
        batch = Variable(batch, volatile=True)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
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
    preds = nn.Softmax()(preds).numpy()
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

    scores = np.concatenate(scores, 0)
    return np.mean(scores), np.std(scores)


def _compute_statistics_of_path(path, model, batch_size, splits, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))

        imgs = np.array([imread(str(fn)).astype(np.float32) for fn in files])

        # Bring images to shape (B, 3, H, W)
        imgs = imgs.transpose((0, 3, 1, 2))
        # Rescale images to be between 0 and 1
        imgs /= 255

        m, s = calculate_inception_score(imgs, model, batch_size, splits, dims, cuda)
    return m, s


def calculate_is_given_path(path, batch_size, splits, cuda, dims):
    if not os.path.exists(path):
        raise RuntimeError('Invalid path: %s' % path)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()
    m, s = _compute_statistics_of_path(path, model, batch_size, splits, dims, cuda)
    return m, s



