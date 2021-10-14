# -*-coding:utf-8-*-
# pytorch --> cifar_model
# Link:
# Data: 2021/6/9 下午4:04
# Author: bronchow
# cifar 神经网络
# --------------------------------------

# 搭建神经网络
import torch
from torch import nn

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


if __name__ == '__main__':
    input = torch.rand((64, 3, 32, 32))
    # print(input.shape)
    my_net = MyCifar()
    print(my_net)
    out = my_net(input)
    print(out.shape)
