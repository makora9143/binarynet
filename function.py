# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch.autograd import Function

def where(cond, x1, x2):
    return cond.float() * x1 + (1 - cond.float()) * x2

class BinaryLinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        weight_b = where(weight>=0, 1, -1)
        output = input.mm(weight_b.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        weight_b = where(weight>=0, 1, -1)
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight_b)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias

class BinaryStraightThroughFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = where(input>=0, 1, -1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        grad_input = grad_output.clone()
        grad_input = grad_input * where(torch.abs(input[0]) <= 1, 1, 0)
        return grad_input


binary_linear = BinaryLinearFunction.apply
bst = BinaryStraightThroughFunction.apply
