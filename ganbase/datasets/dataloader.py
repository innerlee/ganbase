import torch
import torchvision.transforms as transforms
from torchvision.datasets import *
from .looploader import LoopLoader
from .transform import CropBox

# add UnalignedDataset
import os
import random
from PIL import Image
from torch.utils.data import Dataset


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


class UnalignedDataset2(Dataset):
    def __init__(self, rootdirA, rootdirB, transfrom, serial_batch=False):
        super(UnalignedDataset2, self).__init__()
        self.size_A, self.names_A, self.paths_A = walk_all_files_with_suffix(rootdirA)
        self.size_B, self.names_B, self.paths_B = walk_all_files_with_suffix(rootdirB)
        self.transform = transfrom
        self.serial_batch = serial_batch

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
        return A, B, path_A, path_B

    def __len__(self):
        return max(self.size_A, self.size_B)


def loaddata(dataset,
             dataroot,
             imageSize,
             bs,
             nSample=0,
             nWorkers=2,
             pinMemory=True,
             droplast=False,
             loadSize=72,
             datatarget=''):
    """
    load dataset

    returns dataset, dataloader

    args:
    bs: number or list / tuple
    """
    dst = None
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
            root=dataroot,
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
    elif dataset == 'cyclegan':
        dst = UnalignedDataset2(
            rootdirA=dataroot,
            rootdirB=datatarget,
            transfrom=transforms.Compose([
                transforms.Resize(loadSize),
                transforms.CenterCrop(imageSize),
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

    return dst, dataloader, nSample
