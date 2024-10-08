from argparse import ArgumentParser
from models import modelpool
from os import environ
from pathlib import Path
from preprocess import datapool
from torchinfo import summary
from utils import train, seed_all

import torch

def main():
    parser = ArgumentParser(description='PyTorch Training')
    parser.add_argument(
        'weights_file',
        help = 'path to network weights file'
    )
    parser.add_argument(
        '-b', '--batch_size', default=200,
        type=int,metavar='N',help='mini-batch size'
    )
    parser.add_argument(
        '--seed', default=42,
        type=int,help='seed for initializing training.'
    )
    parser.add_argument(
        '-data', '--dataset',
        default='cifar100',type=str,help='dataset'
    )
    parser.add_argument(
        '--net', required = True, type = str,
        help = 'network'
    )
    parser.add_argument('--device', default='0', type=str, help='device')
    parser.add_argument('--time', type=int, required = True)
    args = parser.parse_args()
    seed_all(args.seed)
    environ["CUDA_VISIBLE_DEVICES"] = args.device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, l_te = datapool(args.dataset, args.batch_size)

    net = modelpool(args.net, args.dataset, args.time, 8)
    summary(net)

    state_dict = torch.load(
        args.weights_file,
        weights_only = True,
        map_location='cpu'
    )
    net.load_state_dict(state_dict)
    net = net.to(dev)

    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        n = len(l_te)
        for i, (x, y) in enumerate(l_te):
            x = x.to(dev)
            yh = net(x)
            if args.time > 0:
                yh = yh.sum(0)
            n_el = y.size(0)
            n_corr = (yh.argmax(1) == y).sum().item()
            total += n_el
            correct += n_corr
            print("%4d/%4d acc %5.3f" % (i, n, n_corr / n_el))
        final_acc = 100 * correct / total
    print(final_acc)

if __name__ == "__main__":
    main()
