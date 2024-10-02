from pathlib import Path
from preprocess.augment import Cutout, CIFAR10Policy
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision.transforms import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

DATA_DIR = Path('/tmp/data')

def GetCifar10(batch_size, attack=False):
    trans_t = Compose([
        RandomCrop(32, padding=4),
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
    d_tr = CIFAR10(DATA_DIR, train=True, transform=trans_t, download=True)
    d_te = CIFAR10(DATA_DIR, train=False, transform=trans, download=True)
    l_tr = DataLoader(d_tr, batch_size=batch_size, shuffle=True, num_workers=8)
    l_te = DataLoader(d_te, batch_size=batch_size, shuffle=False, num_workers=8)
    return l_tr, l_te

def GetCifar100(batch_size):
    norm = Normalize(
        mean = [n/255. for n in [129.3, 124.1, 112.4]],
        std = [n/255. for n in [68.2,  65.4,  70.4]]
    )
    trans_t = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        CIFAR10Policy(),
        ToTensor(),
        norm,
        Cutout(n_holes=1, length=16)
    ])
    trans = Compose([ToTensor(), norm])
    d_tr = CIFAR100(
        DATA_DIR, train=True,
        transform=trans_t, download=True
    )
    d_te = CIFAR100(
        DATA_DIR, train=False,
        transform=trans, download=True
    )
    l_tr = DataLoader(
        d_tr, batch_size=batch_size,
        shuffle=True, num_workers=8,
        pin_memory=True
    )
    l_te = DataLoader(
        d_te, batch_size=batch_size,
        shuffle=False,
        drop_last = True
    )
    return l_tr, l_te

def GetImageNet(batch_size):
    trans_t = Compose([
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    trans = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    d_tr = ImageFolder(root=DATA_DIR / 'train', transform=trans_t)
    train_sampler = DistributedSampler(d_tr)
    l_tr = DataLoader(
        d_tr, batch_size=batch_size, shuffle=False,
        num_workers=8,
        sampler=train_sampler, pin_memory=True
    )

    d_te = ImageFolder(root=DATA_DIR / 'val', transform=trans)
    test_sampler = DistributedSampler(d_te)
    l_te = DataLoader(d_te, batch_size=batch_size, shuffle=False, num_workers=2, sampler=test_sampler)
    return l_tr, l_te
