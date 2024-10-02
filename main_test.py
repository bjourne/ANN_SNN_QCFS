import torch

from argparse import ArgumentParser
from models import modelpool
from os import environ
from pathlib import Path
from preprocess import datapool
from torchinfo import summary
from utils import train, val, seed_all

def main():
    parser = ArgumentParser(description='PyTorch Training')
    parser.add_argument(
        'weights_file',
        help = 'path to network weights file'
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
        '-data', '--dataset',
        default='cifar100',type=str,help='dataset'
    )
    parser.add_argument(
        '--net', default = 'vgg16',
        type=str, help = 'network'
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

    net = modelpool(args.net, args.dataset, args.time, 8)
    state_dict = torch.load(
        args.weights_file,
        weights_only = True,
        map_location='cpu'
    )
    net.load_state_dict(state_dict)

    net = net.to(dev)
    acc = val(net, l_te, dev, args.time)
    print(acc)

if __name__ == "__main__":
    main()
