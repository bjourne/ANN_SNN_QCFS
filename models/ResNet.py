"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
from matplotlib.pyplot import xlim
from models.layer import *
from torch.nn import *

import torch.nn as nn


class BasicBlock(Module):
    """Basic Block for resnet 18 and resnet 34
    """
    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, n_in, n_out, stride, T, L):
        super().__init__()
        self.residual_function = Sequential(
            Conv2d(
                n_in, n_out,
                kernel_size=3, stride=stride, padding=1, bias=False
            ),
            BatchNorm2d(n_out),
            IF(T, L),
            Conv2d(
                n_out, n_out * BasicBlock.expansion,
                kernel_size=3, padding=1, bias=False
            ),
            BatchNorm2d(n_out * BasicBlock.expansion)
        )
        self.shortcut = Sequential()
        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or n_in != BasicBlock.expansion * n_out:
            self.shortcut = nn.Sequential(
                Conv2d(n_in, n_out * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(n_out * BasicBlock.expansion)
            )
        self.act = IF(T, L)

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        return self.act(x)

class ResNet(Module):
    def __init__(self, block, num_block, n_classes, T, L):
        super().__init__()
        self.in_channels = 64
        self.T = T
        # self.merge = MergeTemporalDim(0)
        # self.expand = ExpandTemporalDim(0)
        self.conv1 = Sequential(
            Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(64),
            IF()
        )
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, T, L)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, T, L)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, T, L)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, T, L)
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * block.expansion, n_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, T, L):
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(
                self.in_channels, out_channels, stride,
                T, L
            ))
            self.in_channels = out_channels * block.expansion
        return Sequential(*layers)

    def forward(self, x):
        if self.T > 0:
            bs = x.shape[0]
            x.unsqueeze_(1)
            x = x.repeat(self.T, 1, 1, 1, 1)
            x = x.flatten(0, 1).contiguous()
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.T > 0:
            _, n_cls = x.shape
            x = x.view((self.T, bs, n_cls))
        return x

class ResNet4Cifar(nn.Module):
    def __init__(self, block, num_block, n_classes=10):
        super().__init__()
        self.in_channels = 16
        self.T = 0
        self.merge = MergeTemporalDim(0)
        self.expand = ExpandTemporalDim(0)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            IF())
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 16, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 32, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 64, num_block[2], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, n_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def set_T(self, T):
        self.T = T
        for module in self.modules():
            if isinstance(module, (IF, ExpandTemporalDim)):
                module.T = T
            if isinstance(module, IF):
                print(module.thresh)
        return

    def set_L(self, L):
        for module in self.modules():
            if isinstance(module, IF):
                module.L = L
        return

    def forward(self, x):
        if self.T > 0:
            x = add_dimention(x, self.T)
            x = self.merge(x)
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        if self.T > 0:
            output = self.expand(output)
        return output

def resnet18(n_classes, T, L):
    return ResNet(BasicBlock, [2, 2, 2, 2], n_classes, T, L)

def resnet20(n_classes=10):
    return ResNet4Cifar(BasicBlock, [3, 3, 3], n_classes=n_classes)

def resnet34(n_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], n_classes=n_classes)
