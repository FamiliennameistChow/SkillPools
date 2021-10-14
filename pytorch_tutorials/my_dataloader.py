#
# pytorch 入门 -- > 如何使用dataloader
# https://www.bilibili.com/video/BV1hE411t7RN?p=15
# Date: 20210607

# 如何使用torch vision 里给定的数据集
# ----------------------------------------------
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

test_set = torchvision.datasets.CIFAR10(root="./cifar_dataset", train=False, transform=transforms.ToTensor(), download=True)

test_loader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
# DataLoader常用参数
# shuffle 每个epoch数据都会随机重置
# num_workers 执行线程
# drop_last 如果dataset数量 除以 batch_size除不尽, 是否舍弃最后的数据
#

img, target = test_set[0]
print(img.shape)
print(target)

# 取dataloader中的每一个数据, 其中每个数据都是以batch_size进行打包的，

writer = SummaryWriter("dataloader_logs")
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("dataloader", imgs, step)  #注意是add_images
    step += 1

    # print(imgs.shape)
    # print(targets)
    # imgs.shape --> torch.Size([4, 3, 32, 32]) 四张图片
    # targets --> tensor([7, 1, 0, 2])　四个target

writer.close()