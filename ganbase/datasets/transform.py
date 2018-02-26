import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch


class CropBox(object):
    """
    crop and resize
    """

    def __init__(self, i, j, w, h):
        #region yapf: disable
        self.i = i
        self.j = j
        self.w = w
        self.h = h
        #region yapf: enable

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        return img.crop((self.i, self.j, self.i + self.w, self.j + self.h))


class NoiseAugmentor(object):
    """
    add noise (0 for all channels)
    """

    def __init__(self, ratio=0.1):
        self.ratio = ratio
        assert ratio >= 0 and ratio <= 1

    def __call__(self, imgs):
        """
        Args:
            imgs: batch of images Tensor
        Returns:
            Tensor: images with noise
        """
        n, _, w, h = imgs.size()
        mask = np.random.choice(
            [0, 1], (n, 1, w, h), p=[self.ratio,
                                     1 - self.ratio]).astype('float32')
        mask_gpu = torch.from_numpy(mask).cuda(async=True)
        return imgs * mask_gpu


class JitterAugmentor(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size of (0.64 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        crop: crop area ratio from
        interpolation: Default: PIL.Image.BILINEAR
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self,
                 crop=0.64,
                 brightness=0,
                 contrast=0,
                 saturation=0,
                 hue=0,
                 interpolation=Image.BILINEAR):
        #region yapf: disable
        self.crop          = crop
        self.interpolation = interpolation
        self.brightness    = brightness
        self.contrast      = contrast
        self.saturation    = saturation
        self.hue           = hue
        #region yapf: enable
        raise NotImplementedError()

    def __call__(self, imgs):
        """
        Args:
            imgs: batch of images Tensor
        Returns:
            Tensor: images with jitter
        """
