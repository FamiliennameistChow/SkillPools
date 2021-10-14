#
# pytorch 入门 -- > SEQUENTIAL
# https://www.bilibili.com/video/BV1hE411t7RN?p=22
# Date: 20210608

# 1. 手写一个神经网络
# 2. loss function
# 3. 优化器
# ----------------------------------------------

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./cifar_dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)


class MyCifra(nn.Module):
    def __init__(self):
        super(MyCifra, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        # self.pool1 = nn.MaxPool2d(kernel_size=2)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        # self.pool2 = nn.MaxPool2d(kernel_size=2)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # self.pool3 = nn.MaxPool2d(kernel_size=2)
        # self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(1024, 64)
        # self.linear2 = nn.Linear(64, 10)

        self.model1 = nn.Sequential(
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
        # x = self.conv1(x)
        # x = self.pool1(x)
        # x = self.conv2(x)
        # x = self.pool2(x)
        # x = self.conv3(x)
        # x = self.pool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # out = self.linear2(x)

        out = self.model1(x)
        return out


input = torch.rand((64, 3, 32, 32))
# print(input.shape)
my_net = MyCifra()
# print(my_net)
# out = my_net(input)
# print(out.shape)

writer = SummaryWriter("model_logs")
writer.add_graph(my_net, input)


loss_cross = nn.CrossEntropyLoss()

optim = torch.optim.SGD(my_net.parameters(), lr=0.001)

for epoch in range(30):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = my_net(imgs)
        loss = loss_cross(outputs, targets)
        # print("loss", loss)
        optim.zero_grad()  # 上轮梯度清零
        loss.backward()  # 计算本轮的梯度
        optim.step()  # 优化
        running_loss += loss
    writer.add_scalar("loss", running_loss, epoch)
    print("epoch %d --> loss %f" %(epoch, running_loss))

writer.close()
