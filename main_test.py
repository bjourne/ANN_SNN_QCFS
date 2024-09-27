import argparse
import os
import torch
import warnings

from models import modelpool
from os import environ
from pathlib import Path
from preprocess import datapool
from utils import train, val, seed_all, get_logger
from models.layer import *

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting
parser.add_argument(
    '-j', '--workers',
    default = 4,
    type = int,
    metavar = 'N',
    help = 'number of data loading workers'
)
parser.add_argument(
    '-b','--batch_size',default=200, type=int,metavar='N',help='mini-batch size'
)
parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')

def main():
    parser.add_argument(
        '-data', '--dataset',
        default='cifar100',type=str,help='dataset'
    )
    parser.add_argument(
        '-arch','--model',default='vgg16',
        type=str,help='network'
    )
    parser.add_argument(
        '-id', '--identifier',
        type=str,help='model statedict identifier'
    )
    parser.add_argument(
        '-dev','--device',default='0',type=str,help='device'
    )
    parser.add_argument(
        '-T', '--time',
        default=0, type=int, help='snn simulation time'
    )

    args = parser.parse_args()
    seed_all(args.seed)
    environ["CUDA_VISIBLE_DEVICES"] = args.device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, l_te = datapool(args.dataset, args.batch_size)
    net = modelpool(args.model, args.dataset)

    dir = Path('%s-checkpoints' % args.dataset)
    path = dir / (args.identifier + '.pth')
    state_dict = torch.load(path, map_location=torch.device('cpu'))

    # # if old version state_dict
    # keys = list(state_dict.keys())
    # print('keys', keys)
    # for k in keys:
    #     if "relu.up" in k:
    #         print('here!')
    #         state_dict[k[:-7]+'act.thresh'] = state_dict.pop(k)
    #     elif "up" in k:
    #         print('here!')
    #         state_dict[k[:-2]+'thresh'] = state_dict.pop(k)

    net.load_state_dict(state_dict)

    net = net.to(dev)
    net.set_T(args.time)
    net.set_L(8)

    for m in net.modules():
        if isinstance(m, IF):
            print(m.thresh)

    acc = val(net, l_te, dev, args.time)
    print(acc)

if __name__ == "__main__":
    main()
