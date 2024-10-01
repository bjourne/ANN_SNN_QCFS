from cv2 import mean
from sympy import print_rcode
from torch.autograd import Function
from torch.nn import Module, Parameter

import torch

class MergeTemporalDim(Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq):
        return x_seq.flatten(0, 1).contiguous()

class ExpandTemporalDim(Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq):
        y_shape = [self.T, int(x_seq.shape[0]/self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)

class GradFloor(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

myfloor = GradFloor.apply

class IF(Module):
    def __init__(self, T=0, L=8, thresh=8.0):
        super(IF, self).__init__()
        self.thresh = Parameter(torch.tensor([thresh]), requires_grad=True)
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.L = L
        self.T = T
        self.loss = 0

    def forward(self, x):
        if self.T > 0:
            thre = self.thresh.data
            x = self.expand(x)

            mem = 0.5 * thre
            spike_pot = []
            for t in range(self.T):
                mem = mem + x[t, ...]
                spike = (mem - thre >= 0).float() * thre
                mem = mem - spike
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
        else:
            x = x / self.thresh
            x = torch.clamp(x, 0, 1)
            x = myfloor(x * self.L+0.5)/self.L
            x = x * self.thresh
        return x

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1, 1)
    return x
