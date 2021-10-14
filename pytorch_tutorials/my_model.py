#
# pytorch 入门 -- > 如何搭建神经网络
# https://www.bilibili.com/video/BV1hE411t7RN?p=16
# Date: 20210607

# 使用nn model 搭建神经网络
# ----------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()

    def forward(self, input):
        output = input + 1
        return output


# ---------------------------------

net = myModel()   # init
x = torch.tensor(1.0)
output = net(x)
print(output)
