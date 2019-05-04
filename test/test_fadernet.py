# -*- coding: utf-8 -*-


import os
import argparse
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.image
from torch.utils.data import DataLoader
import utils
import models

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag. use 0 or 1")


def attr_flag(s):
    """
    Parse attributes parameters.
    """
    if s == "*":
        return s
    attr = s.split(',')
    assert len(attr) == len(set(attr))
    attributes = []
    for x in attr:
        if '.' not in x:
            attributes.append((x, 2))
        else:
            split = x.split('.')
            assert len(split) == 2 and len(split[0]) > 0
            assert split[1].isdigit() and int(split[1]) >= 2
            attributes.append((split[0], int(split[1])))
    return sorted(attributes, key=lambda x: (x[1], x[0]))


# parse parameters
parser = argparse.ArgumentParser(description='Attributes swapping')

parser.add_argument('--test_dir', default='', type=str)
parser.add_argument('--celeba_attr_path', default='', type=str)
parser.add_argument('--selected_attrs', nargs='+', type=str)
parser.add_argument("--model_path", type=str, default="",
                    help="Trained model path")
parser.add_argument("--n_images", type=int, default=10,
                    help="Number of images to modify")
parser.add_argument("--offset", type=int, default=0,
                    help="First image index")
parser.add_argument("--n_interpolations", type=int, default=10,
                    help="Number of interpolations per image")
parser.add_argument("--alpha_min", type=float, default=1,
                    help="Min interpolation value")
parser.add_argument("--alpha_max", type=float, default=1,
                    help="Max interpolation value")
parser.add_argument("--plot_size", type=int, default=5,
                    help="Size of images in the grid")
parser.add_argument("--row_wise", type=bool_flag, default=True,
                    help="Represent image interpolations horizontally")
parser.add_argument("--output_path", type=str, default="output.png",
                    help="Output path")
parser.add_argument("--img_sz", type=int, default=256, help="Image sizes (images have to be squared)")
parser.add_argument("--img_fm", type=int, default=3, help="Number of feature maps (1 for grayscale, 3 for RGB)")
parser.add_argument("--attr", type=attr_flag, default="Male", help="Attributes to classify")
parser.add_argument("--instance_norm", type=bool_flag, default=False,
                    help="Use instance normalization instead of batch normalization")
parser.add_argument("--init_fm", type=int, default=32, help="Number of initial filters in the encoder")
parser.add_argument("--max_fm", type=int, default=512, help="Number maximum of filters in the autoencoder")
parser.add_argument("--n_layers", type=int, default=6, help="Number of layers in the encoder / decoder")
parser.add_argument("--n_skip", type=int, default=0, help="Number of skip connections")
parser.add_argument("--deconv_method", type=str, default="convtranspose", help="Deconvolution method")
parser.add_argument("--hid_dim", type=int, default=512,
                    help="Last hidden layer dimension for discriminator / classifier")
parser.add_argument("--dec_dropout", type=float, default=0., help="Dropout in the decoder")
params = parser.parse_args()

# check parameters
assert os.path.isfile(params.model_path)
assert params.n_images >= 1 and params.n_interpolations >= 2

# create logger / load trained model
params.n_attr = sum([n_cat for _, n_cat in params.attr])
ae = models.fadernet.AutoEncoder(params)
ae.load_state_dict(torch.load(params.model_path)['netG'])
ae.eval()

# restore main parameters
params.debug = True
params.batch_size = 32
params.v_flip = False
params.h_flip = False
params.img_sz = ae.img_sz
params.attr = ae.attr
params.n_attr = ae.n_attr
if not (len(params.attr) == 1 and params.n_attr == 2):
    raise Exception("The model must use a single boolean attribute only.")

# load dataset

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
transform_list = []
transform_list.append(transforms.Resize(256))
transform_list.append(transforms.CenterCrop(256))
transform_list.append(transforms.ToTensor())
transform_list.append(transforms.Normalize(mean, std))
transform = transforms.Compose(transform_list)
test_dataset = utils.datasets.CelebA(params.test_dir, params.celeba_attr_path,
                                     params.selected_attrs, transform, 'test')

test_data_size = len(test_dataset)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=params.n_images,
    shuffle=False,
    num_workers=0
)


def get_interpolations(ae, images, attributes, params):
    """
    Reconstruct images / create interpolations
    """
    assert len(images) == len(attributes)
    enc_outputs = ae.encode(images)

    # interpolation values
    alphas = np.linspace(1 - params.alpha_min, params.alpha_max, params.n_interpolations)
    alphas = [torch.FloatTensor([1 - alpha, alpha]) for alpha in alphas]

    # original image / reconstructed image / interpolations
    outputs = []
    outputs.append(images)
    outputs.append(ae.decode(enc_outputs, attributes)[-1])
    for alpha in alphas:
        alpha = alpha.unsqueeze(0).expand((len(images), 2)).cuda()
        outputs.append(ae.decode(enc_outputs, alpha)[-1])

    # return stacked images
    return torch.cat([x.unsqueeze(1) for x in outputs], 1).data.cpu()


interpolations = []

for k in range(0, params.n_images, 100):
    i = params.offset + k
    j = params.offset + min(params.n_images, k + 100)
    for n in range(params.offset):
        data = next(iter(test_loader))
    for m in range(100):
        data = next(iter(test_loader))
    images, labels = data
    attributes = torch.zeros(labels.size(0), params.attr[0][1])
    attributes.scatter_(1, labels.long(), 1)
    use_gpu = True
    if use_gpu:
        images = images.cuda()
        ae = ae.cuda()
        attributes = attributes.cuda()

    interpolations.append(get_interpolations(ae, images, attributes, params))

interpolations = torch.cat(interpolations, 0)
assert interpolations.size() == (params.n_images, 2 + params.n_interpolations,
                                 3, params.img_sz, params.img_sz)


def get_grid(images, row_wise, plot_size=5):
    """
    Create a grid with all images.
    """
    n_images, n_columns, img_fm, img_sz, _ = images.size()
    if not row_wise:
        images = images.transpose(0, 1).contiguous()
    images = images.view(n_images * n_columns, img_fm, img_sz, img_sz)
    images.add_(1).div_(2.0)
    return make_grid(images, nrow=(n_columns if row_wise else n_images))


# generate the grid / save it to a PNG file
grid = get_grid(interpolations, params.row_wise, params.plot_size)
matplotlib.image.imsave(params.output_path, grid.numpy().transpose((1, 2, 0)))
