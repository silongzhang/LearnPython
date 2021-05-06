# -*- coding: utf-8 -*-

from math import pi
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.optim

def PI(num_sample):
    sample = torch.rand(num_sample, 2)
    dist = sample.norm(p = 2, dim = 1)
    ratio = (dist < 1).float().mean()
    p = ratio * 4
    return p

def p48(steps):
    x = torch.tensor([pi / 3,  pi / 6], requires_grad=True)
    f = - ((x.cos() ** 2).sum()) ** 2
    print ('f(x) = {}'.format(f))
    optimizer = torch.optim.SGD([x,], lr=0.1, momentum=0)
    for step in range(steps):
        optimizer.zero_grad()
        f.backward()
        optimizer.step()
        f = - ((x.cos() ** 2).sum()) ** 2
        print ('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), f))

def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

def showHimmelblau():
    x = np.arange(-6, 6, 0.1)
    y = np.arange(-6, 6, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = himmelblau([X, Y])
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z)
    ax.view_init(60, -30)
    ax.set_xlabel('x[0]')
    ax.set_ylabel('x[1]')
    fig.show()

def minHimmelblau(x, steps):
    f = himmelblau(x)
    print ('f(x) = {}'.format(f))
    optimizer = torch.optim.Adam([x,])
    for step in range(steps):
        optimizer.zero_grad()
        f.backward()
        optimizer.step()
        f = himmelblau(x)
        if step % 1000 == 0:
            print ('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), f))






