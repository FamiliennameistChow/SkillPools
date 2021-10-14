#
# pytorch 入门 -- > 线性层
# https://www.bilibili.com/video/BV1hE411t7RN?p=20
# Date: 20210607

# 使用nn model 搭建神经网络
#
# ----------------------------------------------
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader


dataset = torchvision.datasets.CIFAR10(root="./cifar_dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, input):
        out = self.linear1(input)
        return out


my_net = MyModel()
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # out = torch.reshape(imgs, (1, 1, 1, -1))
    out = torch.flatten(imgs)
    print(out.shape)
    out = my_net(out)
    print(out.shape)
