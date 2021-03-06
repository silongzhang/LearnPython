# -*- coding: utf-8 -*-

import sys
import math
from math import pi
import random
import numpy as np
import pandas as pd
from pandas_datareader import wb
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim
import torchvision.datasets
from torchvision.datasets import CIFAR10
import torchvision.transforms
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.utils.data
from torch.utils.data import DataLoader
import gym

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
            print('???{}??????loss = {:g}, W = {}'.format(step, loss, w.tolist()))

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
            print('???{}??????loss = {:g}, W = {}'.format(step, loss, w))

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
                print('???{}??????{}??????loss = {:g}'.format(epoch, idx, loss))

    correct = 0
    total = 0
    for images, labels in test_loader:
        x = images.reshape(-1, 28 * 28)
        preds = fc(x)
        predicted = torch.argmax(preds, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    accuracy = correct / total
    print('???????????????????????????{:.1%}'.format(accuracy))

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
            print('#{}, ?????? = {:g}'.format(step, loss))
            
            output = (pred > 0)
            correct = (output == y.bool())
            n_correct_train = correct[train_start:train_end].sum().item()
            n_correct_test = correct[test_start:test_end].sum().item()
            accuracy_train = n_correct_train / n_total_train
            accuracy_test = n_correct_test / n_total_test
            print('?????????????????? = {}, ?????????????????? = {}'.format(accuracy_train, accuracy_test))

def himmelblauTensor(x):
    return (x[:,0] ** 2 + x[:,1] - 11) ** 2 + (x[:,0] + x[:,1] ** 2 - 7) ** 2

def p105(seed, sample_num, train_num, validate_num, test_num, hidden_features, steps):
    torch.manual_seed(seed=seed) # ?????????????????????,?????????????????????????????????
    features = torch.rand(sample_num, 2)  * 12 - 6 # ????????????
    noises = torch.randn(sample_num)
    hims = himmelblauTensor(features) * 0.01
    labels = hims + noises # ????????????
    
    train_mse = torch.mean(noises[:train_num] ** 2)
    validate_mse = torch.mean(noises[train_num:-test_num] ** 2)
    test_mse = torch.mean(noises[-test_num:] ** 2)
    print ('??????:?????????MSE = {:g}, ?????????MSE = {:g}, ?????????MSE = {:g}'.format(
            train_mse, validate_mse, test_mse))

    layers = [nn.Linear(2, hidden_features[0]),]
    for idx, hidden_feature in enumerate(hidden_features):
        layers.append(nn.Sigmoid())
        next_hidden_feature = hidden_features[idx + 1] \
                if idx + 1 < len(hidden_features) else 1
        layers.append(nn.Linear(hidden_feature, next_hidden_feature))
    net = nn.Sequential(*layers) # ??????????????????
    print('??????????????? {}'.format(net))

    optimizer = torch.optim.Adam(net.parameters())
    criterion = nn.MSELoss()

    for step in range(steps):
        outputs = net(features)
        preds = outputs[:, 0]

        loss_train = criterion(preds[:train_num],
                labels[:train_num])
        loss_validate = criterion(preds[train_num:-test_num],
                labels[train_num:-test_num])
        if step % 10000 == 0:
            print ('#{} ?????????MSE = {:g}, ?????????MSE = {:g}'.format(
                    step, loss_train, loss_validate))
    
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    
    print ('?????????MSE = {:g}, ?????????MSE = {:g}'.format(loss_train, loss_validate))
    
    outputs = net(features)
    preds = outputs.squeeze()
    loss = criterion(preds[-test_num:], labels[-test_num:])
    print(loss)

def p135(batch_size, num_epochs):
    # ????????????
    train_dataset = torchvision.datasets.MNIST(root='./data/mnist',
            train=True, transform=torchvision.transforms.ToTensor(),
            download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data/mnist',
            train=False, transform=torchvision.transforms.ToTensor(),
            download=True)
    
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size)
    
    # ??????????????????
    class Net135(torch.nn.Module):
        
        def __init__(self):
            super(Net135, self).__init__()
            self.conv1 = torch.nn.Sequential(
                    torch.nn.Conv2d(1, 64, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(stride=2, kernel_size=2))
            self.dense = torch.nn.Sequential(
                    torch.nn.Linear(128 * 14 * 14, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=0.5),
                    torch.nn.Linear(1024, 10))
            
        def forward(self, x):
            x = self.conv1(x)
            x = x.view(-1, 128 * 14 * 14)
            x = self.dense(x)
            return x
    
    net = Net135()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters()) 
    
    # ??????
    for epoch in range(num_epochs):
        for idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            preds = net(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            
            if idx % 100 == 0:
                print('epoch {}, batch {}, ?????? = {:g}'.format(
                        epoch, idx, loss.item()))
    
    # ??????
    correct = 0
    total = 0
    for images, labels in test_loader:
        preds = net(images)
        predicted = torch.argmax(preds, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    accuracy = correct / total
    print('?????????????????????: {:.1%}'.format(accuracy))

def p148(steps):
    countries = ['BR', 'CA', 'CN', 'FR', 'DE', 'IN', 'IL', 'JP', 'SA', 'GB', 'US',]
    dat = wb.download(indicator='NY.GDP.PCAP.KD',
            country=countries, start=1970, end=2016)
    df = dat.unstack().T
    df.index = df.index.droplevel(0).astype(int)

    class Net(torch.nn.Module):
        
        def __init__(self, input_size, hidden_size):
            super(Net, self).__init__()
            self.rnn = torch.nn.LSTM(input_size, hidden_size)
            self.fc = torch.nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            x = x[:, :, None]
            x, _ = self.rnn(x)
            x = self.fc(x)
            x = x[:, :, 0]
            return x
    
    net = Net(input_size=1, hidden_size=5)

    # ???????????????
    df_scaled = df / df.loc[2000]
    
    # ???????????????????????????
    years = df.index
    train_seq_len = sum((years >= 1971) & (years <= 2000))
    test_seq_len = sum(years > 2000)
    print ('??????????????? = {}, ??????????????? = {}'.format(
            train_seq_len, test_seq_len))
    
    # ????????????????????????????????????
    inputs = torch.tensor(df_scaled.iloc[:-1].values, dtype=torch.float32)
    labels = torch.tensor(df_scaled.iloc[1:].values, dtype=torch.float32)
    
    # ????????????
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    for step in range(steps):
        if step:
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        preds = net(inputs)
        train_preds = preds[:train_seq_len]
        train_labels = labels[:train_seq_len]
        train_loss = criterion(train_preds, train_labels)
        
        test_preds = preds[train_seq_len:]
        test_labels = labels[train_seq_len:]
        test_loss = criterion(test_preds, test_labels)
        
        if step % 500 == 0:
            print ('???{}?????????: loss (?????????) = {}, loss (?????????) = {}'.format(
                    step, train_loss, test_loss))

    preds = net(inputs)
    df_pred_scaled = pd.DataFrame(preds.detach().numpy(),
            index=years[1:], columns=df.columns)
    df_pred = df_pred_scaled * df.loc[2000]
    print(df_pred.loc[2001:])

def p162(epoch_num):
    dataset = CIFAR10(root='./data', download=True,
            transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # ??????????????????
    latent_size = 64 # ????????????
    n_channel = 3 # ???????????????
    n_g_feature = 64 # ???????????????????????????
    gnet = nn.Sequential( 
            # ???????????? = (64, 1, 1)
            nn.ConvTranspose2d(latent_size, 4 * n_g_feature, kernel_size=4,
                    bias=False),
            nn.BatchNorm2d(4 * n_g_feature),
            nn.ReLU(),
            # ?????? = (256, 4, 4)
            nn.ConvTranspose2d(4 * n_g_feature, 2 * n_g_feature, kernel_size=4,
                    stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * n_g_feature),
            nn.ReLU(),
            # ?????? = (128, 8, 8)
            nn.ConvTranspose2d(2 * n_g_feature, n_g_feature, kernel_size=4,
                    stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_g_feature),
            nn.ReLU(),
            # ?????? = (64, 16, 16)
            nn.ConvTranspose2d(n_g_feature, n_channel, kernel_size=4,
                    stride=2, padding=1),
            nn.Sigmoid(),
            # ???????????? = (3, 32, 32)
            )
    print (gnet)
    
    # ??????????????????
    n_d_feature = 64 # ???????????????????????????
    dnet = nn.Sequential( 
            # ???????????? = (3, 32, 32)
            nn.Conv2d(n_channel, n_d_feature, kernel_size=4,
                    stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # ?????? = (64, 16, 16)
            nn.Conv2d(n_d_feature, 2 * n_d_feature, kernel_size=4,
                    stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * n_d_feature),
            nn.LeakyReLU(0.2),
            # ?????? = (128, 8, 8)
            nn.Conv2d(2 * n_d_feature, 4 * n_d_feature, kernel_size=4,
                    stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * n_d_feature),
            nn.LeakyReLU(0.2),
            # ?????? = (256, 4, 4)
            nn.Conv2d(4 * n_d_feature, 1, kernel_size=4),
            # ???????????????????????? = (1, 1, 1)
            )
    print(dnet)

    def weights_init(m): # ?????????????????????????????????
        if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
            init.xavier_normal_(m.weight)
        elif type(m) == nn.BatchNorm2d:
            init.normal_(m.weight, 1.0, 0.02)
            init.constant_(m.bias, 0)
    
    gnet.apply(weights_init)
    dnet.apply(weights_init)

    # ??????
    criterion = nn.BCEWithLogitsLoss()
    
    # ?????????
    goptimizer = torch.optim.Adam(gnet.parameters(),
            lr=0.0002, betas=(0.5, 0.999))
    doptimizer = torch.optim.Adam(dnet.parameters(), 
            lr=0.0002, betas=(0.5, 0.999))
    
    # ???????????????????????????,????????????????????????????????????????????????????????????????????????
    batch_size = 64
    fixed_noises = torch.randn(batch_size, latent_size, 1, 1)
    
    # ????????????
    for epoch in range(epoch_num):
        for batch_idx, data in enumerate(dataloader):
            # ?????????????????????
            real_images, _ = data
            batch_size = real_images.size(0)
            
            # ??????????????????
            labels = torch.ones(batch_size) # ???????????????????????????1
            preds = dnet(real_images) # ???????????????????????????
            outputs = preds.reshape(-1)
            dloss_real = criterion(outputs, labels) # ??????????????????????????????
#            dmean_real = outputs.sigmoid().mean()
            dmean_real = (outputs.sigmoid() >= 0.5).float().mean()
                    # ??????????????????????????????????????????????????????,?????????????????????
            
            noises = torch.randn(batch_size, latent_size, 1, 1) # ????????????
            fake_images = gnet(noises) # ???????????????
            labels = torch.zeros(batch_size) # ????????????????????????0
            fake = fake_images.detach()
                    # ?????????????????????????????????????????????,???????????????????????????.????????????????????????
            preds = dnet(fake) # ????????????????????????
            outputs = preds.view(-1)
            dloss_fake = criterion(outputs, labels) # ???????????????????????????
#            dmean_fake = outputs.sigmoid().mean()
            dmean_fake = (outputs.sigmoid() >= 0.5).float().mean()
                    # ??????????????????????????????????????????????????????,?????????????????????
            
            dloss = dloss_real + dloss_fake # ?????????????????????
            dnet.zero_grad()
            dloss.backward()
            doptimizer.step()
            
            # ??????????????????
            labels = torch.ones(batch_size)
                    # ???????????????????????????????????????????????????????????????
            preds = dnet(fake_images) # ??????????????????????????????
            outputs = preds.view(-1)
            gloss = criterion(outputs, labels) # ????????????????????????
#            gmean_fake = outputs.sigmoid().mean()
            gmean_fake = (outputs.sigmoid() >= 0.5).float().mean()
                    # ??????????????????????????????????????????????????????,?????????????????????
            gnet.zero_grad()
            gloss.backward()
            goptimizer.step()
            
            # ????????????????????????
            print('[{}/{}]'.format(epoch, epoch_num) +
                    '[{}/{}]'.format(batch_idx, len(dataloader)) +
                    '??????????????????:{:g} ??????????????????:{:g}'.format(dloss, gloss) +
                    '?????????????????????:{:g} ?????????????????????:{:g}/{:g}'.format(
                    dmean_real, dmean_fake, gmean_fake))
            if batch_idx % 100 == 0:
                fake = gnet(fixed_noises) # ????????????????????????????????????
                save_image(fake, # ???????????????
                        './data/cifar-10/images_epoch{:02d}_batch{:03d}.png'.format(
                        epoch, batch_idx))

def p177(batch_size, pool_size, test_episodes):
    env = gym.make('CartPole-v0')
    
    model = nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, env.action_space.n)
        )

    def act(model, state, epsilon):
        if random.random() > epsilon: # ????????????
            state = torch.FloatTensor(state).unsqueeze(0)
            q_value = model.forward(state)
            action = q_value.max(1)[1].item()
        else: # ?????????
            action = random.randrange(env.action_space.n)
        return action

    def calc_epsilon(t, epsilon_start=1.0,
        epsilon_final=0.01, epsilon_decay=500):
        epsilon = epsilon_final + (epsilon_start - epsilon_final) \
                * math.exp(-1. * t / epsilon_decay)
        return epsilon

    class ReplayBuffer:
        def __init__(self, capacity):
            self.buffer = deque(maxlen=capacity)
        
        def push(self, state, action, reward, next_state, done):
            state = np.expand_dims(state, 0)
            next_state = np.expand_dims(next_state, 0)
            self.buffer.append((state, action, reward, next_state, done))
        
        def sample(self, batch_size):
            state, action, reward, next_state, done = zip( \
                    *random.sample(self.buffer, batch_size))
            concat_state = np.concatenate(state)
            concat_next_state = np.concatenate(next_state)
            return concat_state, action, reward, concat_next_state, done
        
        def __len__(self):
            return len(self.buffer)
    
    replay_buffer = ReplayBuffer(pool_size)

    optimizer = torch.optim.Adam(model.parameters())
    gamma = 0.99
    episode_rewards = [] # ????????????,??????????????????????????????
    t = 0 # ????????????,????????????epsilon
    
    while True:
        # ??????????????????
        state = env.reset()
        episode_reward = 0

        while True:
            env.render()
            epsilon = calc_epsilon(t)
            action = act(model, state, epsilon)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
    
            state = next_state
            episode_reward += reward
    
            if len(replay_buffer) > batch_size:
                # ????????????????????????
                sample_state, sample_action, sample_reward, sample_next_state, \
                        sample_done = replay_buffer.sample(batch_size)
    
                sample_state = torch.tensor(sample_state, dtype=torch.float32)
                sample_action = torch.tensor(sample_action, dtype=torch.int64)
                sample_reward = torch.tensor(sample_reward, dtype=torch.float32)
                sample_next_state = torch.tensor(sample_next_state,
                        dtype=torch.float32)
                sample_done = torch.tensor(sample_done, dtype=torch.float32)
                
                next_qs = model(sample_next_state)
                next_q, _ = next_qs.max(1)
                expected_q = sample_reward + gamma * next_q * (1 - sample_done)
                
                qs = model(sample_state)
                q = qs.gather(1, sample_action.unsqueeze(1)).squeeze(1)
                
                td_error = expected_q - q
                
                # ?????? MSE ??????
                loss = td_error.pow(2).mean() 
                
                # ????????????????????????
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t += 1
                
            if done: # ????????????
                i_episode = len(episode_rewards)
                print ('???{}????????? = {}'.format(i_episode, episode_reward))
                episode_rewards.append(episode_reward)
                break

        if len(episode_rewards) > 20 and np.mean(episode_rewards[-20:]) > 195:
            break # ????????????

    for test_episode in range(test_episodes):
        observation = env.reset()
        episode_reward = 0
        while True:
            env.render()
            action = act(model, observation, 0)
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
            state = observation
            if done:
                break
        print ('???{}????????? = {}'.format(test_episode, episode_reward))

    env.close()



