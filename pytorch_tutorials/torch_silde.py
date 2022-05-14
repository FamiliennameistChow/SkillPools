# -*-coding:utf-8-*-
# pytorch --> torch_slide.py
# Data: 2022/4/28 下午3:29
# Author: bronchow
# 这里需要使用torch tensor构建buffer
# 测试 数据 save and sample
# --------------------------------------

import torch
import numpy as np

a = torch.ones((3, 4, 4))

print(a)
print(a.shape)
print(a.dtype)

b = torch.ones(3, 5)

state = (a, b)

print(state[0])

buffer = torch.zeros((5, 3, 4, 4))

# save data
buffer[0] = a

# print(buffer)
# print(buffer.shape)

# choice
choice = np.random.choice(5, 2, replace=True)

batch = buffer[choice]

# print(batch)
# print(batch.shape)