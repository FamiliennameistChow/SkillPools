#
# pytorch 入门 -- > 非线性激活
# https://www.bilibili.com/video/BV1hE411t7RN?p=20
# Date: 20210607

# 使用nn model 搭建神经网络
#
# ----------------------------------------------

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

x = torch.tensor([[1, -0.5],
                  [-1, 3]])

x = torch.reshape(x, (-1, 1, 2, 2))
print(x.shape)

dataset = torchvision.datasets.CIFAR10(root="./cifar_dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.sigmod = nn.Sigmoid()

    def forward(self, input):
        out = self.sigmod(input)
        return out


my_net = MyModel()
print(my_net)

out = my_net(x)
print(out)


writer = SummaryWriter("sig_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    out = my_net(imgs)
    writer.add_images("in", imgs, step)
    writer.add_images("out", out, step)
    step += 1

writer.close()
