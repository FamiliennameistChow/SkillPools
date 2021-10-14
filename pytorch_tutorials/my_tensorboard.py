#
# pytorch 入门 -- > tensorboard 数据可视化
# https://www.bilibili.com/video/BV1hE411t7RN?p=8
# Date: 20210527

# ----------------------------------------------


from torch.utils.tensorboard import SummaryWriter
import cv2

writer = SummaryWriter("logs")  #

img = cv2.imread("hymenoptera_data/train/ants/0013035.jpg")
img2 = cv2.imread("hymenoptera_data/train/bees/16838648_415acd9e3f.jpg")
writer.add_image("img", img, 1, dataformats='HWC')
writer.add_image("img", img2, 2, dataformats='HWC')
# y = x

for i in range(100):
    writer.add_scalar("y=x", i, i,)

# writer.add_scalar()

writer.close()