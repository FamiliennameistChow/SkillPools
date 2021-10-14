#
# pytorch 入门 -- > pytorch 加载数据初识
# https://www.bilibili.com/video/BV1hE411t7RN?p=6
# Date: 20210523
# Dateset 整体数据集 1.如何获取每一个数据 2. 总共有多少数据
# Dataloader 为网络提供不同的数据形式
# ----------------------------------------------
import torch

from torch.utils.data import Dataset
import os
import cv2


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path_list = os.listdir(self.path)

    def __getitem__(self, idx):
        image_name = self.img_path_list[idx]
        image_dir = os.path.join(self.root_dir, self.label_dir, image_name)
        img = cv2.imread(image_dir)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path_list)


dir = os.getcwd()
train_dir = os.path.join(dir, "hymenoptera_data", "train")
ants_dataset = MyData(train_dir, "ants")
bees_dataset = MyData(train_dir, "bees")
img, label = bees_dataset[0]

cv2.imshow("img", img)
cv2.waitKey(0)

train_dataset = ants_dataset + bees_dataset
