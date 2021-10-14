#
# pytorch 入门 -- > dataset 与　torchvision的联合使用
# https://www.bilibili.com/video/BV1hE411t7RN?p=14
# Date: 20210607

# 如何使用torch vision 里给定的数据集
# ----------------------------------------------

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset_tran = transforms.Compose([
    transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./cifar_dataset", train=True, transform=dataset_tran, download=True)
test_set = torchvision.datasets.CIFAR10(root="./cifar_dataset", train=False, transform=dataset_tran, download=True)

# print(train_set[0])

# (<PIL.Image.Image image mode=RGB size=32x32 at 0x7F1733B62AC0>, 6)
# 图片, 标签
#
# img, target = train_set[0]
# print(img)
# print(target)
# print(train_set.classes[target])

writer = SummaryWriter("cifar_logs")

for i in range(10):
    img, target = train_set[i]
    writer.add_image("test_img", img, i)

writer.close()