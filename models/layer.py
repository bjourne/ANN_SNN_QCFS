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

# T=0 and L=8 is default
class IF(Module):
    def __init__(self, T, L, thresh=8.0):
        super(IF, self).__init__()
        self.thresh = Parameter(torch.tensor([thresh]), requires_grad=True)
        self.L = L
        self.T = T

    def forward(self, x):
        if self.T > 0:
            thr = self.thresh.data
            if self.mem is None:
                self.mem = torch.zeros_like(x) + thr * 0.5

            self.mem += x
            spks = (self.mem - thr >= 0).float() * thr
            self.mem -= spks
            return spks
        x = x / self.thresh
        x = torch.clamp(x, 0, 1)
        x = myfloor(x * self.L+0.5)/self.L
        return x * self.thresh

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1, 1)
    return x
