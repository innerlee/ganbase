import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
import torch.nn as nn

import ganbase as gb


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
        model = gb.DCGAN_G(imageSize, nc, nz, widthG, nExtraLayerG, nExtraConvG, activationG, normalizeG)
    elif model_choice == 'began':
        model = gb.BEGAN_G(imageSize, nc, nz, widthG, 'elu', 'none')
    elif model_choice == 'sagan':
        model = gb.SAGAN_G(imageSize, nz, widthG)
    else:
        raise NotImplementedError('{} is not supported yet'.format(model_choice))
    print(model)
    # model.load_state_dict(state_dict)
    if use_cuda:
        model = model.cuda()
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.FloatTensor)
    model = nn.DataParallel(model)
    model.load_state_dict(state_dict)
    if use_cuda:
        model = model.cuda()
    z = torch.randn(nImages, nz)
    # z = torch.rand(num_images, z_dim) * 2 - 1 # uniform between [0, 1] if using this
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
            # print(output.shape)
            cv2.imwrite(imagePath, output)


if __name__ == '__main__':
    # 需要设置model_path, 设置保存文件夹的路径，然后取消注释
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    use_cuda = True
    model_choice = 'sagan'
    model_path = '/data00/lihuaxia/models/ganbase/lxh_gan_pth/sagan_celeba.pth'
    output_dir = '/data00/lihuaxia/models/ganbase/results/sagan_celeba'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    generate_images(model_choice=model_choice, model_path=model_path, output_dir=output_dir,
                    imageSize=64, nc=3, nz=128, widthG=64, use_cuda=use_cuda, nImages=50000)
