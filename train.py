# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from torchvision import datasets, transforms

from tqdm import tqdm

import model
from weight_clip import weight_clip

def train(args): 
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    train_loader = data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
                       batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = data.DataLoader(
        datasets.MNIST('./data', train=False,
                       transform=transforms.ToTensor()),
                       batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.binary == 'connect':
        net = model.BinaryConnect(args.in_features, args.out_features)
    else:
        net = model.BinaryNet(args.in_features, args.out_features)
    # net = nn.DataParallel(net)
    print(net)

    if args.cuda:
        net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    creterion = nn.NLLLoss()

    for epoch in range(1, args.epochs+1):
        train_epoch(epoch, net, creterion, optimizer, train_loader, args)
        test_epoch(net, creterion, test_loader, args)

def train_epoch(epoch, net, creterion, optimizer, train_loader, args, valid_data=None):
    losses = 0
    accs = 0
    net.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader), 1):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data.view(args.batch_size, -1)), Variable(target)

        optimizer.zero_grad()

        output = net(data)
        loss = creterion(output, target)
        loss.backward()
        optimizer.step()
        weight_clip(net.parameters())

        y_pred = torch.max(output, 1)[1]
        accs += (torch.mean((y_pred == target).float())).data[0]

        losses += loss.data[0]
    print("Epoch {0}: Train Loss={1:.3f}, Train Accuracy={2:.3f}".format(epoch, losses / batch_idx, accs / batch_idx))

    if valid_data is not None:
        pass

def test_epoch(net, creterion, test_loader, args):
    net.eval()
    losses = 0
    accs = 0
    for batch_idx, (data, target) in enumerate(test_loader, 1):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data.view(args.test_batch_size, -1)), Variable(target)

        output = net(data)
        loss = creterion(output, target)

        y_pred = torch.max(output, 1)[1]
        accs += (torch.mean((y_pred == target).float())).data[0]

        losses += loss.data[0]
    print("\tTest Loss={0:.3f}, Test Accuracy={1:.3f}".format(losses / batch_idx, accs / batch_idx))

