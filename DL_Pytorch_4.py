# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision
import torchvision.utils as vutils
from torch.utils import data
import torchvision.transforms as transforms
from torchvision import utils
from torchvision import datasets
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

class TestDataset(data.Dataset):#继承Dataset
    def __init__(self):
        self.Data=np.asarray([[1,2],[3,4],[2,1],[3,4],[4,5]])#一些由2维向量表示的数据集
        self.Label=np.asarray([0,1,0,1,2])#这是数据集对应的标签

    def __getitem__(self, index):
        #把numpy转换为Tensor
        txt=torch.from_numpy(self.Data[index])
        label=torch.tensor(self.Label[index])
        return txt,label 

    def __len__(self):
        return len(self.Data)

def p65():
    Test=TestDataset()
    print(Test[2])  #相当于调用__getitem__(2)
    print(Test.__len__())

    test_loader = data.DataLoader(Test,batch_size=2,shuffle=False)
    for i,traindata in enumerate(test_loader):
        print('i:',i)
        Data,Label=traindata
        print('data:',Data)
        print('Label:',Label)

    dataiter=iter(test_loader)
    imgs,labels=next(dataiter)

def p67():
    transforms.Compose([
        #将给定的 PIL.Image 进行中心切割，得到给定的 size，
        #size 可以是 tuple，(target_height, target_width)。
        #size 也可以是一个 Integer，在这种情况下，切出来的图片形状是正方形。            
        transforms.CenterCrop(10),
        #切割中心点的位置随机选取
        transforms.RandomCrop(20, padding=0),
        #把一个取值范围是 [0, 255] 的 PIL.Image 或者 shape 为 (H, W, C) 的 numpy.ndarray，
        #转换为形状为 (C, H, W)，取值范围是 [0, 1] 的 torch.FloatTensor
        transforms.ToTensor(),
        #规范化到[-1,1]
        transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
    ])

    my_trans=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    train_data = datasets.ImageFolder('./data/torchvision_data', transform=my_trans)
    train_loader = data.DataLoader(train_data,batch_size=8,shuffle=True,)

    for i_batch, img in enumerate(train_loader):
        if i_batch == 0:
            print(img[1])
            fig = plt.figure()
            grid = utils.make_grid(img[0])
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.show()
            utils.save_image(grid,'test01.png')
        break

    Image.open('test01.png')

def p71():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
            self.bn = nn.BatchNorm2d(20)
    
        def forward(self, x):
            x = F.max_pool2d(self.conv1(x), 2)
            x = F.relu(x) + F.relu(-x)
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = self.bn(x)
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            x = F.softmax(x, dim=1)
            return x

    #定义输入
    input = torch.rand(32, 1, 28, 28)
    #实例化神经网络
    model = Net()
    #将model保存为graph
    with SummaryWriter(log_dir='logs',comment='Net') as w:
        w.add_graph(model, (input, ))

def p72():
    input_size = 1
    output_size = 1
    num_epoches = 60
    learning_rate = 0.01
    
    dtype = torch.FloatTensor
    writer = SummaryWriter(log_dir='logs',comment='Linear')
    np.random.seed(100) 
    x_train = np.linspace(-1, 1, 100).reshape(100,1) 
    y_train = 3*np.power(x_train, 2) +2+ 0.2*np.random.rand(x_train.size).reshape(100,1) 
    
    
    model = nn.Linear(input_size, output_size)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epoches):
        inputs = torch.from_numpy(x_train).type(dtype)
        targets = torch.from_numpy(y_train).type(dtype)
    
        output = model(inputs)
        loss = criterion(output, targets)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 保存loss的数据与epoch数值
        writer.add_scalar('训练损失值', loss, epoch)

def p73():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class CNNNet(nn.Module):
        def __init__(self):
            super(CNNNet,self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
            self.conv2 = nn.Conv2d(in_channels=16,out_channels=36,kernel_size=3,stride=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(1296,128)
            self.fc2 = nn.Linear(128,10)      

        def forward(self,x):
            x=self.pool1(F.relu(self.conv1(x)))
            x=self.pool2(F.relu(self.conv2(x)))
            #print(x.shape)
            x=x.view(-1,36*6*6)
            x=F.relu(self.fc2(F.relu(self.fc1(x))))
            return x

    net = CNNNet()
    net=net.to(device)

    LR=0.001
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    #初始化数据
    for m in net.modules():
        if isinstance(m,nn.Conv2d):
            nn.init.normal_(m.weight)
            nn.init.xavier_normal_(m.weight)
            nn.init.kaiming_normal_(m.weight)#卷积层参数初始化
            nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Linear):
            nn.init.normal_(m.weight)#全连接层参数初始化

    #训练模型
    for epoch in range(2):
        running_loss = 0.0
        for i, dt in enumerate(trainloader, 0):
            # 获取训练数据
            inputs, labels = dt
            inputs, labels = inputs.to(device), labels.to(device)

            # 权重参数梯度清零
            optimizer.zero_grad()
    
            # 正向及反向传播
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # 显示损失值
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    
    print('Finished Training')
    
    writer = SummaryWriter(log_dir='logs',comment='feature map')

    for i, dt in enumerate(trainloader, 0):
            # 获取训练数据
            inputs, labels = dt
            inputs, labels = inputs.to(device), labels.to(device)
            x=inputs[0].unsqueeze(0)
            break
    
    img_grid = vutils.make_grid(x, normalize=True, scale_each=True, nrow=2)

    net.eval()
    for name, layer in net._modules.items():
        # 为fc层预处理x
        x = x.view(x.size(0), -1) if "fc" in name else x
        print(x.size())
    
        x = layer(x)
        print(f'{name}')
    
        # 查看卷积层的特征图
        if  'layer' in name or 'conv' in name:
            x1 = x.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
            img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=4)  # normalize进行归一化处理
            writer.add_image(f'{name}_feature_maps', img_grid, global_step=0)

