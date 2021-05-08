# -*- coding: utf-8 -*-

from math import pi
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.optim
import torchvision.datasets
import torchvision.transforms
import torch.utils.data

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

def p66(steps):
    x = torch.tensor([[1., 1., 1.], [2., 3., 1.], 
            [3., 5., 1.], [4., 2., 1.], [5., 4., 1.]])
    y = torch.tensor([-10., 12., 14., 16., 18.])
    w = torch.zeros(3, requires_grad=True)
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([w,],)

    for step in range(steps):
        pred = torch.mv(x, w)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            print('step = {}, loss = {:g}, W = {}'.format(step, loss, w.tolist()))

def p67(steps):
    x = torch.tensor([[1., 1.], [2., 3.], [3., 5.], [4., 2.], [5., 4.]])
    y = torch.tensor([-10., 12., 14., 16., 18.]).reshape(-1, 1)

    fc = torch.nn.Linear(2, 1)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(fc.parameters())
    weights, bias = fc.parameters()

    for step in range(steps):
        pred = fc(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            print('step = {}, loss = {:g}, weights = {}, bias={}'.\
                  format(step, loss, weights[0, :].tolist(), bias.item()))

def p80(steps):
    x = torch.tensor([[1., 1., 1.], [2., 3., 1.],
        [3., 5., 1.], [4., 2., 1.], [5., 4., 1.]])
    y = torch.tensor([0., 1., 1., 0., 1.])
    w = torch.zeros(3, requires_grad=True)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam([w,],)
    
    for step in range(steps):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pred = torch.mv(x, w)
        loss = criterion(pred, y)
        if step % 10000 == 0:
            print('第{}步：loss = {:g}, W = {}'.format(step, loss, w.tolist()))

def p84(steps):
    x = torch.tensor([[1., 1., 1.], [2., 3., 1.],
        [3., 5., 1.], [4., 2., 1.], [5., 4., 1.]])
    y = torch.tensor([0, 2, 1, 0, 2])
    w = torch.zeros(3, 3, requires_grad=True)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([w,],)
    
    for step in range(steps):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pred = torch.mm(x, w)
        loss = criterion(pred, y)
        if step % 10000 == 0:
            print('第{}步：loss = {:g}, W = {}'.format(step, loss, w))

def p85(batch_size, num_epochs):
    train_dataset = torchvision.datasets.MNIST(root='./data/mnist',
        train=True, transform=torchvision.transforms.ToTensor(),
        download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data/mnist',
        train=False, transform=torchvision.transforms.ToTensor(),
        download=True)
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size)
    print('len(train_loader) = {}'.format(len(train_loader)))
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size)
    print('len(test_loader) = {}'.format(len(test_loader)))
    
    for images, labels in train_loader:
        print ('image.size() = {}'.format(images.size()))
        print ('labels.size() = {}'.format(labels.size()))
        break

    plt.imshow(images[0, 0], cmap='gray')
    plt.title('label = {}'.format(labels[0]))
    
    fc = torch.nn.Linear(28 * 28, 10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fc.parameters())
    
    for epoch in range(num_epochs):
        for idx, (images, labels) in enumerate(train_loader):
            x = images.reshape(-1, 28*28)
            
            optimizer.zero_grad()
            preds = fc(x)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            if idx % 100 == 0:
                print('第{}趟第{}批：loss = {:g}'.format(epoch, idx, loss))

    correct = 0
    total = 0
    for images, labels in test_loader:
        x = images.reshape(-1, 28 * 28)
        preds = fc(x)
        predicted = torch.argmax(preds, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    accuracy = correct / total
    print('测试集上的准确率：{:.1%}'.format(accuracy))

def p89(steps):
    url = 'http://raw.githubusercontent.com/zhiqingxiao/pytorch-book/master/chapter06_logistic/FB.csv'
    df = pd.read_csv(url, index_col=0)
    train_start, train_end = sum(df.index >= '2017'), sum(df.index >= '2013')
    test_start, test_end = sum(df.index >= '2018'), sum(df.index >= '2017')
    n_total_train = train_end - train_start
    n_total_test = test_end - test_start
    s_mean = df[train_start:train_end].mean()
    s_std = df[train_start:train_end].std()
    n_features = 5
    df_feature = ((df - s_mean) / s_std).iloc[:, :n_features]
    s_label = (df['Volume'] < df['Volume'].shift(1)).astype(int)
    
    fc = torch.nn.Linear(n_features, 1)
    weights, bias = fc.parameters()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(fc.parameters())
    
    x = torch.tensor(df_feature.values, dtype=torch.float32)
    y = torch.tensor(s_label.values.reshape(-1, 1), dtype=torch.float32)
    
    for step in range(steps):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pred = fc(x)
        loss = criterion(pred[train_start:train_end], y[train_start:train_end])
        
        if step % 500 == 0:
            print('#{}, 损失 = {:g}'.format(step, loss))
            
            output = (pred > 0)
            correct = (output == y.bool())
            n_correct_train = correct[train_start:train_end].sum().item()
            n_correct_test = correct[test_start:test_end].sum().item()
            accuracy_train = n_correct_train / n_total_train
            accuracy_test = n_correct_test / n_total_test
            print('训练集准确率 = {}, 测试集准确率 = {}'.format(accuracy_train, accuracy_test))



