from textwrap import fill

from torchvision.datasets import CIFAR100
from torchvision.transforms import *
from torchvision import datasets
from torch.utils.data import DataLoader
import torch
import os
from preprocess.augment import Cutout, CIFAR10Policy

# your own data dir
DIR = {'CIFAR10': '~/datasets', 'CIFAR100': '~/datasets', 'ImageNet': 'YOUR_IMAGENET_DIR'}

def GetCifar10(batchsize, attack=False):
    trans_t = Compose([RandomCrop(32, padding=4),
                                  RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  ToTensor(),
                                  Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  Cutout(n_holes=1, length=16)
                                  ])
    if attack:
        trans = Compose([ToTensor()])
    else:
        trans = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans, download=True)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader

def GetCifar100(batchsize):
    trans_t = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        CIFAR10Policy(),
        ToTensor(),
        Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
        Cutout(n_holes=1, length=16)
    ])
    trans = Compose([
        ToTensor(),
        Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])
    ])
    train_data = CIFAR100(DIR['CIFAR100'], train=True, transform=trans_t, download=True)
    test_data = CIFAR100(DIR['CIFAR100'], train=False, transform=trans, download=True)
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(
        test_data, batch_size=batchsize,
        shuffle=False, num_workers=4,
        drop_last = True
    )
    return train_dataloader, test_dataloader

def GetImageNet(batchsize):
    trans_t = Compose([RandomResizedCrop(224),
                                RandomHorizontalFlip(),
                                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])

    trans = Compose([Resize(256),
                            CenterCrop(224),
                            ToTensor(),
                            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    train_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'train'), transform=trans_t)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader =DataLoader(train_data, batch_size=batchsize, shuffle=False, num_workers=8, sampler=train_sampler, pin_memory=True)

    test_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'val'), transform=trans)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=2, sampler=test_sampler)
    return train_dataloader, test_dataloader
