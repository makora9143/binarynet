# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import


def weight_clip(params, low=-1.0, high=1.0):
    for p in params:
        p.data.clamp_(low, high)


