# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import


import argparse
import train

parser = argparse.ArgumentParser(description='Binary Neural Networks')
parser.add_argument('--binary', type=str, default='connect',
        help='BinaryConnect or BinaryNet')
parser.add_argument('--cuda', type=bool, default=False,
        help='Use cuda or not')
parser.add_argument('--in_features', type=int, default=784,
        help='input features dim')
parser.add_argument('--out_features', type=int, default=10,
        help='output features dim')
parser.add_argument('--batch_size', type=int, default=100,
        help='batch size')
parser.add_argument('--test_batch_size', type=int, default=1000,
        help='batch size')
parser.add_argument('--lr', type=float, default=0.001,
        help='Learning rate')
parser.add_argument('--epochs', type=int, default=20,
        help='Epochs')
args = parser.parse_args()

train.train(args)
