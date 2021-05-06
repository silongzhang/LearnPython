# -*- coding: utf-8 -*-

import torch

def PI(num_sample):
    sample = torch.rand(num_sample, 2)
    dist = sample.norm(p = 2, dim = 1)
    ratio = (dist < 1).float().mean()
    pi = ratio * 4
    return pi




