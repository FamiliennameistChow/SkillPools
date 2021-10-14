# -*-coding:utf-8-*-
# pytorch 入门 -- > how to save and load mode
# https://www.bilibili.com/video/BV1hE411t7RN?p=27
# Date: 20210609
# Author: bornchow
# ---------------------------------
import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
#
# # save model 1  save model structure and weights
# torch.save(vgg16, "vgg16_method1.pth")
#
#
# # load model 1
# vgg16_load1 = torch.load("vgg16_method1.pth")
# # print(vgg16_load1)

# # save model 2 save model weights
# torch.save(vgg16.state_dict(), "vgg16_weights.pth")
#
# # load model 2
# vgg16 = torchvision.models.vgg16(pretrained=False)
# vgg16.load_state_dict(torch.load("vgg16_weights.pth"))
# # vgg16_load2 = torch.load("vgg16_method2.pth")
# print(vgg16)


# 陷阱 save model 1
class MyCifra(nn.Module):
    def __init__(self):
        super(MyCifra, self).__init__()
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
        out = self.model1(x)
        return out


my_net = MyCifra()
torch.save(my_net, "my_net.pth")

# load my net  需要将模型的定义写出来
model = torch.load("my_net.pth")
