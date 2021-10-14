#
# pytorch 入门 -- > 卷积
# https://www.bilibili.com/video/BV1hE411t7RN?p=16
# Date: 20210607

# 使用nn model 搭建神经网络
# ----------------------------------------------

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./cifar_dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        y = self.conv1(x)
        return y


my_net = MyModel()
print(my_net)

setp = 0
writer = SummaryWriter("dataloader_logs")
for data in dataloader:
    imgs, target = data
    output = my_net(imgs)
    writer.add_images("input", imgs, setp)
    # torch.Size([64, 6, 30, 30]) --> [xxx, 3, 30, 30]
    output1 = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output1, setp)
    print(output.shape)
    print(imgs.shape)
    setp += 1

writer.close()
