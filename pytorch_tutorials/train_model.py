# -*-coding:utf-8-*-
# train_model
# Link: https://www.bilibili.com/video/BV1hE411t7RN?p=28
# Data: 2021/6/9 下午3:13
# Author: bronchow
# --------------------------------------
import math

import torch.optim.optimizer
import torchvision
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from cifar_model import MyCifar

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
class MyCifar(nn.Module):
    def __init__(self):
        super(MyCifar, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        out = self.model(x)
        return out


net = MyCifar()

# loss function
loss_fn = nn.CrossEntropyLoss()

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

for epoch in range(epochs):
    total_train_loss = 0.0
    net.train()  # if they are affected, e.g. Dropout, BatchNorm,
    for data in train_dataloader:
        img, target = data
        output = net(img)
        loss = loss_fn(output, target)
        # 　优化模型
        optimizer.zero_grad()
        loss.backward()
        scheduler.step()

        total_train_step += 1
        total_train_loss += loss
    writer.add_scalar("train loss", total_train_loss / (math.ceil(train_dataset_size / BATCH_SIZE)), epoch)

    # 每训练一轮进行测试
    total_test_loss = 0
    total_acc = 0
    net.eval()  # if they are affected, e.g. Dropout, BatchNorm,
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
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
    # torch.save(net.state_dict(), "./train_model/cifar_weights_model_{}.pth".format(epoch))
    print("model saved")

writer.close()
