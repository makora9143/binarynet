# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from module import BinaryLinear, BinaryStraightThrough

class BinaryConnect(nn.Module):
    def __init__(self, in_features, out_features, num_units=2048):
        super(BinaryConnect, self).__init__()

        self.net = nn.Sequential(
                BinaryLinear(in_features, num_units),
                nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15),
                nn.ReLU(),
                BinaryLinear(num_units, num_units),
                nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15),
                nn.ReLU(),
                BinaryLinear(num_units, num_units),
                nn.BatchNorm1d(num_units, eps=1e-4, momentum=0.15),
                nn.ReLU(),
                BinaryLinear(num_units, out_features),
                nn.BatchNorm1d(out_features, eps=1e-4, momentum=0.15),
                nn.LogSoftmax()
                )

    def forward(self, x):
        return self.net(x)


class BinaryNet(nn.Module):
    def __init__(self, in_features, out_features, num_units=4096):
        super(BinaryNet, self).__init__()

        self.net = nn.Sequential(
                nn.Dropout(p=0.2),
                BinaryLinear(in_features, num_units),
                nn.BatchNorm1d(num_units, eps=1e-4),
                BinaryStraightThrough(),
                nn.Dropout(),
                BinaryLinear(num_units, num_units),
                nn.BatchNorm1d(num_units, eps=1e-4),
                BinaryStraightThrough(),
                nn.Dropout(),
                BinaryLinear(num_units, num_units),
                nn.BatchNorm1d(num_units, eps=1e-4),
                BinaryStraightThrough(),
                nn.Dropout(),
                BinaryLinear(num_units, 10),
                nn.BatchNorm1d(10, eps=1e-4),
                nn.LogSoftmax()
                )

    def forward(self, x):
        return self.net(x)
