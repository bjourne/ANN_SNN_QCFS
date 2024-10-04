import torch.nn as nn

from torch.nn import *
from torch.nn.init import constant_, kaiming_normal_, zeros_
from models.layer import *

cfg = {
    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}

class VGG(Module):
    def __init__(self, vgg_name, n_classes, T, L):
        super(VGG, self).__init__()
        self.n_in = 3
        self.T = T
        self.layer1 = self._make_layers(cfg[vgg_name][0], T, L)
        self.layer2 = self._make_layers(cfg[vgg_name][1], T, L)
        self.layer3 = self._make_layers(cfg[vgg_name][2], T, L)
        self.layer4 = self._make_layers(cfg[vgg_name][3], T, L)
        self.layer5 = self._make_layers(cfg[vgg_name][4], T, L)
        if n_classes == 1000:
            self.classifier = nn.Sequential(
                Flatten(),
                Linear(512*7*7, 4096),
                IF(T, L),
                Dropout(0.0),
                Linear(4096, 4096),
                IF(T, L),
                Dropout(0.0),
                Linear(4096, n_classes)
            )
        else:
            self.classifier = nn.Sequential(
                Flatten(),
                Linear(512, 4096),
                IF(T, L),
                Dropout(0.0),
                Linear(4096, 4096),
                IF(T, L),
                Dropout(0.0),
                Linear(4096, n_classes)
            )
        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                constant_(m.weight, val=1)
                zeros_(m.bias)
            elif isinstance(m, Linear):
                zeros_(m.bias)

    def _make_layers(self, cfg, T, L):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(AvgPool2d(2, 2))
            else:
                layers.append(
                    Conv2d(self.n_in, x, kernel_size=3, padding=1)
                )
                layers.append(BatchNorm2d(x))
                layers.append(IF(T, L))
                layers.append(Dropout(0.0))
                self.n_in = x
        return Sequential(*layers)

    def forward_once(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return self.classifier(x)

    def forward(self, x):
        if self.T > 0:
            for m in self.modules():
                if isinstance(m, IF):
                    m.mem = None
            y = [self.forward_once(x) for _ in range(self.T)]
            return torch.stack(y)
        return self.forward_once(x)

class VGG_woBN(nn.Module):
    def __init__(self, vgg_name, n_classes, dropout):
        super(VGG_woBN, self).__init__()
        self.n_in = 3
        self.T = 0
        self.merge = MergeTemporalDim(0)
        self.expand = ExpandTemporalDim(0)
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)
        if n_classes == 1000:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*7*7, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, n_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                IF(),
                nn.Dropout(dropout),
                nn.Linear(4096, n_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg, dropout, T, L):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.n_in, x, kernel_size=3, padding=1))
                layers.append(IF(T, L))
                layers.append(nn.Dropout(dropout))
                self.n_in = x
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.T > 0:
            x = add_dimention(x, self.T)
            x = self.merge(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        if self.T > 0:
            out = self.expand(out)
        return out

def vgg16(n_classes, T, L):
    return VGG('VGG16', n_classes, T, L)

def vgg16_wobn(n_classes, dropout=0.1):
    return VGG_woBN('VGG16', n_classes, dropout)

def vgg19(n_classes, dropout):
    return VGG('VGG19', n_classes, dropout)
