# -*-coding:utf-8-*-
# pytorch --> train_gpu_1
# Link: https://www.bilibili.com/video/BV1hE411t7RN?p=31
# Data: 2021/6/10 上午10:35
# Author: bronchow
# 使用 GPU训练　方式一：
# 在 网络模型, 数据(输入，标注), 损失函数, 优化器　处 调用.cuda()即可
# --------------------------------------

import math

import torch.optim.optimizer
import torchvision
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from cifar_model import MyCifar
import time


# 参数
BATCH_SIZE = 64

# 准备数据集
train_dataset = torchvision.datasets.CIFAR10("./cifar_dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10("./cifar_dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)

# #数据集长度
train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print("len of train dataset: %d" % train_dataset_size)
print("len of test dataset: %d" % test_dataset_size)

# #dataloader
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# 创建网络模型
net = MyCifar()
if torch.cuda.is_available():
    net = net.cuda()  # 网络模型调用cuda


# loss function
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# optimizer
learning_rate = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

# 设置训练网络的参数
# 训练的次数
total_train_step = 0
# 测试的次数
total_test_step = 0
# 训练的轮数
epochs = 1000
# 添加tensorboard
writer = SummaryWriter("train_logs")

start_time = time.time()
for epoch in range(epochs):
    total_train_loss = 0.0
    net.train()  # if they are affected, e.g. Dropout, BatchNorm,
    for data in train_dataloader:
        img, target = data
        if torch.cuda.is_available():
            img = img.cuda()
            target = target.cuda()
        output = net(img)
        loss = loss_fn(output, target)
        # 　优化模型
        optimizer.zero_grad()
        loss.backward()
        scheduler.step()

        total_train_step += 1
        total_train_loss += loss
    writer.add_scalar("train loss", total_train_loss / (math.ceil(train_dataset_size / BATCH_SIZE)), epoch)
    train_time = time.time()
    print("time: ", train_time-start_time)

    # 每训练一轮进行测试
    total_test_loss = 0
    total_acc = 0
    net.eval()  # if they are affected, e.g. Dropout, BatchNorm,
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss

            # 计算准确率，详看test_acc.py
            acc = (outputs.argmax(1) == targets).sum()
            total_acc += acc

    writer.add_scalar("val loss", total_test_loss / (math.ceil(test_dataset_size / BATCH_SIZE)), epoch)
    writer.add_scalar("test acc", total_acc / test_dataset_size, epoch)
    print("epoch {}/{}  train loss : {} val loss: {} test acc: {}".format(epoch,
                                                                          epochs,
                                                                          total_train_loss / (math.ceil(
                                                                              train_dataset_size / BATCH_SIZE)),
                                                                          total_test_loss / (math.ceil(
                                                                              test_dataset_size / BATCH_SIZE)),
                                                                          total_acc / test_dataset_size))

    torch.save(net, "./train_model/cifar_model_{}.pth".format(epoch))
    torch.save(net.state_dict(), "./train_model/cifar_weights_model_{}.pth".format(epoch))
    print("model saved")
writer.close()
