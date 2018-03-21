#!/usr/bin/env python
import os
import sys
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
sys.path.insert(0, os.path.abspath('..'))
import ganbase as gb  # pylint: disable=C0413


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


print("Preparing Dataset...")
cifar = dset.CIFAR10(
    root='data/',
    download=True,
    transform=transforms.Compose([
        transforms.Scale(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

imgs = IgnoreLabelDataset(cifar)
dataloader = torch.utils.data.DataLoader(imgs, batch_size=50000)

print("Calculating Inception Score...")

for i, batch in enumerate(dataloader, 0):
    result = gb.inception_score(batch, cuda=True, batch_size=32, splits=10)
    print("Inception Score: {}".format(result[0]))
    print("Standard Deviation: {}".format(result[1]))
