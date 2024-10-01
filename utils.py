import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
import numpy as np
import random
import logging

from itertools import islice
from models import IF
from os import environ
from torch.nn.functional import cross_entropy

def seed_all(seed=1029):
    random.seed(seed)
    environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def train(net, device, loader, opt, epoch, n_epochs):
    phase = "train" if net.training else "test"
    args = phase, epoch, n_epochs
    print("== %s %3d/%3d ==" % args)

    tot_loss = 0
    n_tot_el = 0
    n_tot_corr = 0
    n_batches = len(loader)
    for i, (x, y) in enumerate(islice(loader, n_batches)):
        if net.training:
            opt.zero_grad()
        y = y.to(device)
        x = x.to(device)
        yh = net(x)

        loss = cross_entropy(yh, y)
        if net.training:
            loss.backward()
            opt.step()
        loss = loss.item()

        n_el = y.size(0)
        n_corr = (yh.argmax(1) == y).sum().item()

        n_tot_el += n_el
        n_tot_corr += n_corr
        tot_loss += loss
        acc = n_corr / n_el

        print('%4d/%4d, loss/acc: %.4f/%.2f' % (i, n_batches, loss, acc))
    return tot_loss / n_batches, n_tot_corr / n_tot_el


def val(net, loader, device, T):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        n = len(loader)
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            if T > 0:
                yh = net(x).mean(0)
            else:
                yh = net(x)
            _, predicted = yh.cpu().max(1)

            n_batch = y.size(0)
            n_corr = predicted.eq(y).sum().item()

            total += n_batch
            correct += n_corr
            print("%4d/%4d acc %5.3f" % (i, n, n_corr / n_batch))
        final_acc = 100 * correct / total
    return final_acc
