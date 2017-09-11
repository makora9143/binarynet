# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from function import binary_linear, bst

class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BinaryLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0, 1 * (math.sqrt(1. / self.in_features)))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, input):
        if self.bias is not None:
            return binary_linear(input, self.weight, self.bias)
        return binary_linear(input, self.weight)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class BinaryStraightThrough(nn.Module):
    def __init__(self, inplace=False):
        super(BinaryStraightThrough, self).__init__()

    def forward(self, input):
        return bst(input)

    def __repr__(self):
        return self.__class__.__name__ + ' ()'
