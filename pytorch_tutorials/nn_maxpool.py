#
# pytorch 入门 -- > 池化
# https://www.bilibili.com/video/BV1hE411t7RN?p=19
# Date: 20210607

# 使用nn model 搭建神经网络
# 最大池化保留输入的特征，但是减小数据量
# ----------------------------------------------

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./cifar_dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))

print(input.shape)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        out = self.maxpool1(x)
        return out


my_net = MyModel()
print(my_net)

output = my_net(input)
print(output)

writer = SummaryWriter("maxpool_logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = my_net(imgs)
    writer.add_images("in", imgs, step)
    writer.add_images("out", output, step)
    step += 1

writer.close()
