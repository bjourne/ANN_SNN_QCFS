import argparse
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim

from models import modelpool
from os import environ
from preprocess import datapool
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import train, val, seed_all

parser = argparse.ArgumentParser(description='PyTorch Training')
# just use default setting

# 4 workers ~ 2s/batch

parser.add_argument('-j','--workers', default=4, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size', default=300, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed', default=42, type=int, help='seed for initializing training. ')
parser.add_argument('-suffix','--suffix', default='', type=str,help='suffix')
parser.add_argument('-T', '--time', default=0, type=int, help='snn simulation time')

# model configuration
parser.add_argument('-data', '--dataset',default='cifar100',type=str,help='dataset')
parser.add_argument('-arch','--net',default='vgg16',type=str,help='net')

# training configuration
parser.add_argument('--epochs',default=300,type=int,metavar='N',help='number of total epochs to run')
# 0.05 for cifar100 / 0.1 for cifar10
parser.add_argument('-lr','--lr',default=0.1,type=float,metavar='LR', help='initial learning rate')
parser.add_argument('-wd','--weight_decay',default=5e-4, type=float, help='weight_decay')
parser.add_argument('-dev','--device',default='0',type=str,help='device')
parser.add_argument('-L', '--L', default=8, type=int, help='Step L')

args = parser.parse_args()

environ["CUDA_VISIBLE_DEVICES"] = args.device
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    global args
    seed_all(args.seed)
    # preparing data
    l_tr, l_te = datapool(args.dataset, args.batch_size)
    # preparing net
    net = modelpool(args.net, args.dataset)
    net.set_L(args.L)

    net.to(dev)

    opt = SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=args.epochs)
    best_acc = 0

    identifier = args.net

    identifier += '_L[%d]'%(args.L)

    if not args.suffix == '':
        identifier += '_%s'%(args.suffix)

    n_epochs = args.epochs
    for epoch in range(n_epochs):

        net.train()
        tr_loss, tr_acc = train(net, dev, l_tr, opt, epoch, n_epochs)
        scheduler.step()

        print(scheduler.get_last_lr())

        net.eval()
        te_loss, te_acc = train(net, dev, l_te, opt, epoch, n_epochs)

        fmt = 'losses %5.3f/%5.3f acc %5.3f/%5.3f'
        print(fmt % (tr_loss, te_loss, tr_acc, te_acc))

        if best_acc < te_acc:
            print('New best accuracy: %5.3f' % te_acc)
            best_acc = te_acc
            torch.save(net.state_dict(), '%s.pth' % identifier)

main()
