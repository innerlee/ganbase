# -*- coding: utf-8 -*-

import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

import models


def unnormalize_tensor2d(tensor, mean, std):
    if tensor.dim() == 3:
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
    elif tensor.dim() == 4:
        tensor = torch.transpose(tensor, 0, 1)
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        tensor = torch.transpose(tensor, 0, 1)
    return tensor


def generate_images(model_choice, model_path, output_dir, imageSize=64, nc=3, nz=64, widthG=32,
                    nExtraLayerG=0, nExtraConvG=0, activationG='leakyrelu', normalizeG='batch',
                    nImages=1000, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), use_cuda=True):
    state_dict = torch.load(model_path, map_location='cpu')
    if model_choice == 'dcgan':
        model = models.dcgan.Generator(imageSize, nc, nz, widthG, nExtraLayerG, nExtraConvG, activationG, normalizeG)
    elif model_choice == 'began':
        model = models.began.Generator(imageSize, nc, nz, widthG, 'elu', 'none')
    elif model_choice == 'sagan':
        model = models.sagan.Generator(imageSize, nz, widthG)
    else:
        raise NotImplementedError('{} is not supported yet'.format(model_choice))
    model.load_state_dict(state_dict)
    if use_cuda:
        model = model.cuda()
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    # model = nn.DataParallel(model)
    model.load_state_dict(state_dict)
    if use_cuda:
        model = model.cuda()
    # z = torch.randn(nImages, nz)
    z = torch.rand(nImages, nz) * 2 - 1  # uniform between [0, 1] if using this
    with torch.no_grad():
        for i in tqdm(range(nImages)):
            input = torch.unsqueeze(z[i], dim=0)
            if use_cuda:
                input = input.cuda()
            if model_choice in ('dcgan', 'began'):
                output = model(input)
            elif model_choice == 'sagan':
                output, _, _ = model(input)
            output = unnormalize_tensor2d(output, torch.Tensor(mean), torch.Tensor(std)).cpu()
            output = (output.numpy() * 255.0).astype(np.uint8)
            output = np.transpose(np.squeeze(output, axis=0), (1, 2, 0))
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            imagePath = os.path.join(output_dir, model_choice + '_' + str(i).zfill(6) + '.jpg')
            cv2.imwrite(imagePath, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str)
    parser.add_argument('--n_images', type=int)
    parser.add_argument('--model_choice', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()
    if args.gpu:
        use_cuda = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        use_cuda = False
    # nImages = 50000

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
    # sagan celeba/imagenet
    # generate_images(model_choice=model_choice, model_path=model_path, output_dir=output_dir,
    #                 imageSize=64, nc=3, nz=128, widthG=64, use_cuda=use_cuda, nImages=nImages)
    # sagan cifar10
    # generate_images(model_choice=args.model_choice, model_path=args.model_path, output_dir=args.output_path,
    #                 imageSize=32, nc=3, nz=128, widthG=64, use_cuda=use_cuda, nImages=args.n_images)
    # began celeba/imagenet
    # generate_images(model_choice=model_choice, model_path=model_path, output_dir=output_dir,
    #                 imageSize=64, nc=3, nz=64, widthG=128, use_cuda=use_cuda, nImages=10)
    # began cifar10
    # generate_images(model_choice=args.model_choice, model_path=args.model_path, output_dir=args.output_path,
    #                 imageSize=32, nc=3, nz=64, widthG=64, use_cuda=use_cuda, nImages=args.n_images)
    # wgangp/wgan/dcgan celeba/imagenet
    generate_images(model_choice=args.model_choice, model_path=args.model_path, output_dir=args.output_path,
                    imageSize=64, nc=3, nz=64, widthG=64, use_cuda=use_cuda, nImages=args.n_images)
    # wgangp/wgan/dcgan cifar10
    # generate_images(model_choice=args.model_choice, model_path=args.model_path, output_dir=args.output_path,
    #                 imageSize=32, nc=3, nz=64, widthG=64, use_cuda=use_cuda, nImages=args.n_images)
