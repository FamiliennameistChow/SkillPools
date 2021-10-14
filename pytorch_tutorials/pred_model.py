# -*-coding:utf-8-*-
# pytorch --> pred_model
# Link: https://www.bilibili.com/video/BV1hE411t7RN?p=33
# Data: 2021/6/10 下午3:29
# Author: bronchow
# 模型推理
# --------------------------------------
import torch
from torchvision import transforms
from PIL import Image
from cifar_model import MyCifar

img = Image.open("./imgs/airplane.png")

print(img)

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
])

img_tensor = transform(img)
img = torch.reshape(img_tensor, (1, 3, 32, 32))

print(img_tensor.shape)

my_net = torch.load("cifar_model_33.pth", map_location=torch.device('cpu'))
my_net.eval()
with torch.no_grad():
    output = my_net(img)
print(output)
print(output.argmax(1))
