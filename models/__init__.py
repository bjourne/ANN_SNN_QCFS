from .ResNet import *
from .VGG import *
from .layer import *

def modelpool(MODELNAME, DATANAME, T, L):
    if 'imagenet' in DATANAME.lower():
        n_cls = 1000
    elif '100' in DATANAME.lower():
        n_cls = 100
    else:
        n_cls = 10
    if MODELNAME.lower() == 'vgg16':
        return vgg16(n_cls, T, L)
    elif MODELNAME.lower() == 'vgg16_wobn':
        return vgg16_wobn(n_cls)
    elif MODELNAME.lower() == 'resnet18':
        return resnet18(n_cls, T, L)
    elif MODELNAME.lower() == 'resnet34':
        return ResNet([3, 4, 6, 3], n_cls, T, L)
    elif MODELNAME.lower() == 'resnet20':
        return ResNet4Cifar([3, 3, 3], n_cls, T, L)
    assert False
