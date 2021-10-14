#
# pytorch 入门 -- > transforms 基础
# https://www.bilibili.com/video/BV1hE411t7RN?p=10
# Date: 20210604


# transform 用法　-> tensor数据类型
# 注意输入的图像是什么格式 tensor, PIL , numpy
#

# 常用的transfrom
# ----------------------------------------------


from torch.utils.tensorboard import SummaryWriter
import cv2


from torchvision import transforms

writer = SummaryWriter("logs")  #

img = cv2.imread("hymenoptera_data/train/ants/0013035.jpg")
# img2 = cv2.imread("hymenoptera_data/train/bees/16838648_415acd9e3f.jpg")


# To Tensor
tran_tensor = transforms.ToTensor()
img_tensor = tran_tensor(img)
writer.add_image("img_tensor", img_tensor)

# Normalize parma均值，标准差 , 输入必须是tensor类型的图像
# TypeError: tensor is not a torch image.如果格式错误
tran_norma = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_normal = tran_norma(img_tensor)
writer.add_image("img_normal", img_normal)

# resize (h, w)　输入的图像必须是PIL格式 返回的是 PIL格式
tran_resize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
img_resize = tran_resize(img)
writer.add_image("img_resize", img_resize)

# CenterCrop  保留中心(size,size)的区域
tran_centercrop = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(100),
    transforms.ToTensor(),
])

img_cenCrop = tran_centercrop(img)
writer.add_image("img_cenCrop", img_cenCrop)

# randomCrop
tran_randcrop = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(512),
    transforms.ToTensor(),
])

for i in range(10):
    img_randcrop = tran_randcrop(img)
    writer.add_image("img_randcrop", img_randcrop, i)


writer.close()
