#
# pytorch 入门 -- > pretrained model in torchvision
# https://www.bilibili.com/video/BV1hE411t7RN?p=26
# Date: 20210609

# 如何修改torchvision网络模型
# ----------------------------------------------

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

vgg16_fasle = torchvision.models.vgg16(pretrained=False)
vgg16_model = torchvision.models.vgg16(pretrained=False)

vgg16_fasle.add_module("linear2", nn.Linear(64, 10))
vgg16_fasle.classifier.add_module("linear3", nn.Linear(1000, 64))

print(vgg16_fasle)

vgg16_model.classifier[6] = nn.Linear(4096, 10)

print(vgg16_model)