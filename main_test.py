import torch

from argparse import ArgumentParser
from models import modelpool
from os import environ
from pathlib import Path
from preprocess import datapool
from torchinfo import summary
from utils import train, val, seed_all, get_logger
from models.layer import *

def main():
    parser = ArgumentParser(description='PyTorch Training')
    parser.add_argument(
        'weights_file',
        help = 'path to network weights file'
    )
    parser.add_argument(
        '-j', '--workers',
        default = 4,
        type = int,
        metavar = 'N',
        help = 'number of data loading workers'
    )
    parser.add_argument(
        '-b','--batch_size',default=200,
        type=int,metavar='N',help='mini-batch size'
    )
    parser.add_argument(
        '--seed',default=42,
        type=int,help='seed for initializing training.'
    )
    parser.add_argument(
        '-suffix','--suffix',
        default='', type=str,help='suffix'
    )
    parser.add_argument(
        '-data', '--dataset',
        default='cifar100',type=str,help='dataset'
    )
    parser.add_argument(
        '-arch','--model',default='vgg16',
        type=str,help='network'
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
    state_dict = torch.load(
        args.weights_file,
        weights_only = True,
        map_location='cpu'
    )
    net.load_state_dict(state_dict)

    net = net.to(dev)
    net.set_T(args.time)
    net.set_L(8)
    acc = val(net, l_te, dev, args.time)
    print(acc)

if __name__ == "__main__":
    main()
