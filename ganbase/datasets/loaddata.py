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
    return dataset, dataloader
    bs: number or list / tuple
    """
    if dataroot is None:
        dataroot = f'/mnt/SSD/datasets/{dataset}'

    if dataset in ['imagenet', 'folder', 'lfw', 'lfwcrop']:
        # folder dataset
        data = ImageFolder(
            root=dataroot,
            transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.CenterCrop(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    elif dataset == 'celebA':
        data = ImageFolder(
            root=dataroot,
            transform=transforms.Compose([
                CropBox(25, 50, 128, 128),
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    elif dataset == 'lsun':
        data = LSUN(
            db_path=dataroot,
            classes=['bedroom_train'],
            transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.CenterCrop(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    elif dataset == 'cifar10':
        data = CIFAR10(
            root=dataroot,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    elif dataset == 'mnist':
        data = MNIST(
            root=dataroot,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

    assert data

    bs = [bs] if isinstance(bs, int) else list(bs)

    dataloader = []
    if nSample == 0:
        for b in bs:
            dataloader.append(
                LoopLoader(
                    torch.utils.data.DataLoader(
                        data,
                        batch_size=b,
                        pin_memory=pinMemory,
                        shuffle=True,
                        num_workers=int(nWorkers),
                        drop_last=droplast)))
        nSample = len(data)
    else:
        for b in bs:
            dataloader.append(
                LoopLoader(
                    torch.utils.data.DataLoader(
                        data,
                        batch_size=b,
                        pin_memory=pinMemory,
                        shuffle=False,
                        num_workers=int(nWorkers),
                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                            range(nSample)),
                        drop_last=droplast)))

    if len(dataloader) == 1:
        dataloader = dataloader[0]

    return data, dataloader, nSample
