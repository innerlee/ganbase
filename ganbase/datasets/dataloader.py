import torch
import torchvision.transforms as transforms
from torchvision.datasets import *
from .looploader import LoopLoader
from .transform import CropBox


def loaddata(dataset,
             dataroot,
             imageSize,
             bs,
             nSample=0,
             nWorkers=2,
             pinMemory=True,
             droplast=False):
    """
    load dataset

    returns dataset, dataloader

    args:
    bs: number or list / tuple
    """
    if dataset in ['imagenet', 'folder', 'lfw', 'lfwcrop']:
        # folder dataset
        dst = ImageFolder(
            root=dataroot,
            transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.CenterCrop(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    elif dataset == 'celeba-crop':
        dst = ImageFolder(
            root=dataroot,
            transform=transforms.Compose([
                CropBox(25, 50, 128, 128),
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    elif dataset == 'celeba':
        dst = ImageFolder(
            root=dataroot,
            transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.CenterCrop(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    elif dataset == 'lsun':
        dst = LSUN(
            db_path=dataroot,
            classes=['bedroom_train'],
            transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.CenterCrop(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    elif dataset == 'cifar10':
        dst = CIFAR10(
            root=dataroot,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    elif dataset == 'mnist':
        dst = MNIST(
            root=dataroot,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

    assert dst

    bs = [bs] if isinstance(bs, int) else list(bs)

    dataloader = []
    if nSample == 0:
        for b in bs:
            dataloader.append(
                LoopLoader(
                    torch.utils.data.DataLoader(
                        dst,
                        batch_size=b,
                        pin_memory=pinMemory,
                        shuffle=True,
                        num_workers=int(nWorkers),
                        drop_last=droplast)))
        nSample = len(dst)
    else:
        for b in bs:
            dataloader.append(
                LoopLoader(
                    torch.utils.data.DataLoader(
                        dst,
                        batch_size=b,
                        pin_memory=pinMemory,
                        shuffle=False,
                        num_workers=int(nWorkers),
                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                            range(nSample)),
                        drop_last=droplast)))

    if len(dataloader) == 1:
        dataloader = dataloader[0]

    return dst, dataloader
