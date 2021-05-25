# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torch
# 导入 pytorch 内置的 mnist 数据
from torchvision.datasets import mnist 
#import torchvision
#导入预处理模块
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.utils.data import DataLoader
#导入nn及优化器
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from tensorboardX import SummaryWriter

def p52():
    # 定义一些超参数
    train_batch_size = 64
    test_batch_size = 128
    num_epoches = 20

    #定义预处理函数
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])
    #下载数据，并对数据进行预处理
    train_dataset = mnist.MNIST('./data', train=True, transform=transform, download=True)
    test_dataset = mnist.MNIST('./data', train=False, transform=transform)
    #得到一个生成器
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    class Net(nn.Module):
        """
        使用sequential构建网络，Sequential()函数的功能是将网络的层组合到一起
        """
        def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
            super(Net, self).__init__()
            self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),nn.BatchNorm1d(n_hidden_1))
            self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),nn.BatchNorm1d(n_hidden_2))
            self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = self.layer3(x)
            return x

    lr = 0.01
    momentum = 0.9

    #实例化模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(28 * 28, 300, 100, 10)
    model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # 开始训练
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    writer = SummaryWriter(log_dir='logs',comment='train-loss')
    
    for epoch in range(num_epoches):
        train_loss = 0
        train_acc = 0
        model.train()
        #动态修改参数学习率
        if epoch%5==0:
            optimizer.param_groups[0]['lr']*=0.9
            print('lr = {:.4f}'.format(optimizer.param_groups[0]['lr']))
        for img, label in train_loader:
            img=img.to(device)
            label = label.to(device)
            img = img.view(img.size(0), -1)
            # 前向传播
            out = model(img)
            loss = criterion(out, label)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录误差
            train_loss += loss.item()
            # 保存loss的数据与epoch数值
            writer.add_scalar('Train', train_loss/len(train_loader), epoch)
            # 计算分类的准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            train_acc += acc
            
        losses.append(train_loss / len(train_loader))
        acces.append(train_acc / len(train_loader))
        # 在测试集上检验效果
        eval_loss = 0
        eval_acc = 0
        #net.eval() # 将模型改为预测模式
        model.eval()
        for img, label in test_loader:
            img=img.to(device)
            label = label.to(device)
            img = img.view(img.size(0), -1)
            out = model(img)
            loss = criterion(out, label)
            # 记录误差
            eval_loss += loss.item()
            # 记录准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]
            eval_acc += acc
            
        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_acc / len(test_loader))
        print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
              .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader), 
                         eval_loss / len(test_loader), eval_acc / len(test_loader)))

    plt.title('train loss')
    plt.plot(np.arange(len(losses)), losses)
    #plt.plot(np.arange(len(eval_losses)), eval_losses)
    #plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.legend(['Train Loss'], loc='upper right')
    plt.show()

def p60():
    # 超参数
    LR = 0.01
    BATCH_SIZE = 32
    EPOCH = 12

    # 生成训练数据
    # torch.unsqueeze() 的作用是将一维变二维，torch只能处理二维的数据
    x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)  
    # 0.1 * torch.normal(x.size())增加噪点
    y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))
    torch_dataset = Data.TensorDataset(x,y)
    #得到一个代批量的生成器
    loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

    class Net2(torch.nn.Module):
        # 初始化
        def __init__(self):
            super(Net2, self).__init__()
            self.hidden = torch.nn.Linear(1, 20)
            self.predict = torch.nn.Linear(20, 1)
     
        # 前向传递
        def forward(self, x):
            x = F.relu(self.hidden(x))
            x = self.predict(x)
            return x

    net_SGD = Net2()
    net_Momentum = Net2()
    net_RMSProp = Net2()
    net_Adam = Net2()
     
    nets = [net_SGD, net_Momentum, net_RMSProp, net_Adam]

    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.9)
    opt_RMSProp = torch.optim.RMSprop(net_RMSProp.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSProp, opt_Adam]
     
    loss_func = torch.nn.MSELoss()
    loss_his = [[], [], [], []]  # 记录损失
 
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):
            for net, opt,l_his in zip(nets, optimizers, loss_his):
                output = net(batch_x)  # get output for every net
                loss = loss_func(output, batch_y)  # compute loss for every net
                opt.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradients
                opt.step()  # apply gradients
                l_his.append(loss.data.numpy())  # loss recoder

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(loss_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()

