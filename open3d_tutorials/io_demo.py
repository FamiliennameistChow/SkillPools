#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: open3d_tutorials --- > io_demo.py
# Author: bornchow
# Time:20211207
# 本程序演示 点云的读写与 open3d中pointcloud的基本类
# ------------------------------------

import os
import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("./data/bun000.ply")
print(pcd)

