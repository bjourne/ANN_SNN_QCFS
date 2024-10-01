from cv2 import mean
from sympy import print_rcode
from torch.autograd import Function
from torch.nn import Module, Parameter

import torch

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
        self.L = L
        self.T = T

    def forward(self, x):
        if self.T > 0:
            # Expansion
            y_shape = [self.T, int(x.shape[0]/self.T)]
            y_shape.extend(x.shape[1:])
            x = x.view(y_shape)
            thre = self.thresh.data

            mem = 0.5 * thre
            spike_pot = []
            for t in range(self.T):
                mem = mem + x[t, ...]
                spike = (mem - thre >= 0).float() * thre
                mem = mem - spike
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)

            # Contraction
            x = x.flatten(0, 1).contiguous()
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
