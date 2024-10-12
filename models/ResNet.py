"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
from matplotlib.pyplot import xlim
from models.layer import *
from torch.nn import *

class BasicBlock(Module):
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
                n_out, n_out,
                kernel_size=3, padding=1, bias=False
            ),
            BatchNorm2d(n_out)
        )
        self.shortcut = Sequential()
        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or n_in != n_out:
            self.shortcut = Sequential(
                Conv2d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(n_out)
            )
        self.act = IF(T, L)

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        return self.act(x)

def make_layer(n_in, n_out, n_blocks, stride, T, L):
    strides = [stride] + [1] * (n_blocks - 1)
    layers = []
    for stride in strides:
        layers.append(BasicBlock(n_in, n_out, stride, T, L))
        n_in = n_out
    return Sequential(*layers)

class ResNet(Module):
    def __init__(self, num_block, n_classes, T, L):
        super().__init__()
        self.T = T
        self.conv1 = Sequential(
            Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(64),
            IF(T, L)
        )
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = make_layer(64, 64, num_block[0], 1, T, L)
        self.conv3_x = make_layer(64, 128, num_block[1], 2, T, L)
        self.conv4_x = make_layer(128, 256, num_block[2], 2, T, L)
        self.conv5_x = make_layer(256, 512, num_block[3], 2, T, L)

        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512, n_classes)

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x):
        if self.T > 0:
            for m in self.modules():
                if isinstance(m, IF):
                    m.mem = None
            y = [self.forward_once(x) for _ in range(self.T)]
            return torch.stack(y)
        return self.forward_once(x)

class ResNet4Cifar(Module):
    def __init__(self, num_block, n_classes, T, L):
        super().__init__()
        self.in_channels = 16
        self.T = T
        self.conv1 = Sequential(
            Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(16),
            IF(T, L)
        )
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = make_layer(16, 16, num_block[0], 1, T, L)
        self.conv3_x = make_layer(16, 32, num_block[1], 2, T, L)
        self.conv4_x = make_layer(32, 64, num_block[2], 2, T, L)
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(64, n_classes)

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x):
        if self.T > 0:
            for m in self.modules():
                if isinstance(m, IF):
                    m.mem = None
            y = [self.forward_once(x) for _ in range(self.T)]
            return torch.stack(y)
        return self.forward_once(x)

def resnet18(n_cls, T, L):
    return ResNet([2, 2, 2, 2], n_cls, T, L)
