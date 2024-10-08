from .ResNet import *
from .VGG import *
from .layer import *

def modelpool(MODELNAME, DATANAME, T, L):
    if 'imagenet' in DATANAME.lower():
        n_classes = 1000
    elif '100' in DATANAME.lower():
        n_classes = 100
    else:
        n_classes = 10
    if MODELNAME.lower() == 'vgg16':
        return vgg16(n_classes, T, L)
    elif MODELNAME.lower() == 'vgg16_wobn':
        return vgg16_wobn(n_classes=n_classes)
    elif MODELNAME.lower() == 'resnet18':
        return resnet18(n_classes, T, L)
    elif MODELNAME.lower() == 'resnet34':
        return resnet34(n_classes=n_classes)
    elif MODELNAME.lower() == 'resnet20':
        return ResNet4Cifar([3, 3, 3], n_classes, T, L)
    assert False
