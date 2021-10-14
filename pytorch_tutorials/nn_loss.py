#
# pytorch 入门 -- > Loss Function
# https://www.bilibili.com/video/BV1hE411t7RN?p=22
# Date: 20210608

# 1. loss function计算
# ----------------------------------------------
import torch
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)

target = torch.tensor([1, 2, 5], dtype=torch.float32)


inputs = torch.reshape(inputs, (1, 1, 1, 3))

target = torch.reshape(target, (1, 1, 1, 3))

Loss = nn.L1Loss(reduction="sum")

out = Loss(inputs, target)
print(out)

loss_mse = nn.MSELoss()

out2 = loss_mse(inputs, target)

print(out2)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])

x = torch.reshape(x, (1, 3))

loss_cross = nn.CrossEntropyLoss()
out3 = loss_cross(x, y)

print(out3)