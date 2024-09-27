from .getdataloader import *

def datapool(name, batchsize):
    if name == 'cifar10':
        return GetCifar10(batchsize)
    elif name == 'cifar100':
        return GetCifar100(batchsize)
    elif name == 'imagenet':
        return GetImageNet(batchsize)
    assert False
