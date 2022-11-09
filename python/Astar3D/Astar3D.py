#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: python --- > Astar3D.py
# Author: bornchow
# Time:20221010
#
# ------------------------------------
import copy

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource
from matplotlib import cm
import cv2
import math
import time


# f(n) = h(n) + g(n)
# the node for a-star
class Node:
    def __init__(self, x, y, z, father=None):
        self.x = x
        self.y = y
        self.z = z
        self.cost_to_go = 0.0
        self.cost_so_far = 0.0
        self.father = father

    def get_pos(self):
        return self.x, self.y, self.z

    def get_cost_so_far(self):
        return self.cost_so_far

    def get_f(self):
        return self.cost_to_go + self.cost_so_far

    def print_node(self):
        if self.father is not None:
            print(" node:  {}, -> cost: {} ==> father: {}".format(self.get_pos(),
                                                                  self.get_f(),
                                                                  self.father.get_pos()))
        else:
            print(" node:  {}, -> cost: {} ==> father: None".format(self.get_pos(),
                                                                    self.get_f()))


class PathIterator:
    def __init__(self, node):
        self.node = node

    def __next__(self):
        """
        迭代操作
        :return: 当前节点的位置
        """
        if self.node.father is not None:
            self.node = self.node.father
            return self.node.get_pos()
        else:
            raise StopIteration

    def __iter__(self):
        return self


# 优先级队列
class PriorQueue:
    def __init__(self):
        self.list = []

    def is_empty(self):
        return len(self.list) == 0

    def in_list(self, pos):
        for i in range(len(self.list) - 1, -1, -1):
            if self.list[i].get_pos() == pos:
                return self.list[i]
        return None

    def del_node(self, pos):
        for i in range(len(self.list) - 1, -1, -1):
            if self.list[i].get_pos() == pos:
                del self.list[i]

    def push(self, node):
        if self.is_empty():
            self.list.append(node)
        else:
            self.list.append(node)

            for i in range(len(self.list)-1, 0, -1):
                if self.list[i].get_f() > self.list[i-1].get_f():
                    self.list[i], self.list[i-1] = self.list[i-1], self.list[i]

    def pop(self):
        return self.list.pop()

    # return the last node
    def top(self):
        return self.list[-1]

    def print_queue(self):
        print(" this queue ____________ ")
        for i in range(0, len(self.list)):
            self.list[i].print_node()


class Map:
    def __init__(self):
        self.width = 1
        self.height = 1
        self.data = None

    def read_map(self, file_dir):
        """ read map data, resize and smooth
        :param file_dir: dir for map file
        :return:
        """
        print("Loading map from {}".format(file_dir))
        file_in = open(file_dir, 'r')
        new_map = []
        for row in file_in.readlines():
            elevation = [float(point) for point in row.strip().split(' ')]
            new_map.append(elevation)

        # resize map to [100, 100, 500]
        new_map = np.array(new_map)
        new_map = cv2.resize(new_map, (50, 50))
        min = np.min(new_map)
        max = np.max(new_map)
        new_map = new_map * 250 /(max-min) - (min * 250)/(max-min)
        new_map = cv2.GaussianBlur(new_map, (9, 9), 5, 5)

        self.data = new_map
        self.height, self.width = self.data.shape
        print(" Loaded map with size {}".format(self.data.shape))

    def draw_map(self, path=None, waypoints=None):
        """
        :param path: 最终路径点
        :param waypoints: 起始点与终点
        :return:
        """
        print("Drawing map..... ")
        X, Y = np.meshgrid(np.arange(0, self.width, 1),
                           np.arange(0, self.height, 1))
        Z = self.data

        # plt.figure('Surface', facecolor='lightgray')
        fig = plt.figure()
        # ax3d = plt.gca(projection='3d')
        ax3d = fig.add_subplot(projection='3d')
        ax3d.set_xlabel('X', fontsize=14)
        ax3d.set_ylabel('Y', fontsize=14)
        ax3d.set_zlabel('Z', fontsize=14)
        ax3d.set_zlim(0, 800)

        if path is not None:
            p_x, p_y, p_z = [], [], []
            for p in path:
                p_x.append(p[0])
                p_y.append(p[1])
                p_z.append(p[2])
            # ax3d.plot3D(p_x, p_y, p_z, color="blue")  # 线段表示
            ax3d.scatter(p_x, p_y, p_z, s=1, c="blue", alpha=1.0)  # 散点表示

        if waypoints is not None:
            wp_x, wp_y, wp_z = [], [], []
            for p in waypoints:
                wp_x.append(p[0])
                wp_y.append(p[1])
                wp_z.append(p[2])
            ax3d.scatter(wp_x, wp_y, wp_z, s=4, c="red", depthshade=False)

        # 参考 https://qa.1r1g.com/sf/ask/1066092751/ 修改透明度
        # ax3d.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color="grey", alpha=0.8)  # 用网格表示地形
        ax3d.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.gist_earth, alpha=0.85)  # 是曲面表示地形
        # ax3d.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.cool, alpha=0.8)  # 是曲面表示地形

        plt.show()


class Astar:
    def __init__(self, waypoint_dir, map_dir):
        self.terrain_map = Map()
        self.terrain_map.read_map(map_dir)

        # get the waypoint info from txt
        waypoint_file = open(waypoint_dir, 'r')
        self.waypoints = []
        for row in waypoint_file.readlines():
            sg_pair = [int(p) for p in row.strip().split(' ')]
            self.waypoints.append(sg_pair)

        self.waypoints = np.array(self.waypoints)
        print("waypoints: ...........")
        for p in self.waypoints:
            p[2] = p[2] + self.terrain_map.data[p[1]][p[0]]
            print("{} with the terrain elev: {}".format(p, self.terrain_map.data[p[1]][p[0]]))
        print(self.waypoints.shape)
        print("waypoints: ...........")
        # self.terrain_map.draw_map(path=None, waypoints=self.waypoints)

        self.path = []  # the final path
        # start search .....
        for i in range(1, len(self.waypoints)):
            self.search(self.waypoints[i-1], self.waypoints[i])
        self.terrain_map.draw_map(path=self.path, waypoints=self.waypoints)

    def search_nei_node(self, this_node, above, height_mini):
        """ 搜索邻近节点
        :param this_node: cur node
        :param above: 离地的最大距离限制
        :param height_mini 离地最小高度限制
        :return: new pose list
        """
        # 假设是八连通扩展，按z分为三层
        xy_dirs = [0, 1, 0, -1, 1, 1, -1, -1, 0]
        z_dirs = [-1, 0, 1]
        this_pos = this_node.get_pos()

        # 扩展邻近节点，并判断节点的是否超出地图
        pos_list = []
        for z in z_dirs:
            if this_pos[2] + z < self.terrain_map.data[this_pos[1]][this_pos[0]] + height_mini\
                    or this_pos[2] + z > self.terrain_map.data[this_pos[1]][this_pos[0]]+above:
                continue
            for i in range(1, len(xy_dirs)):
                new_pos = (this_pos[0]+xy_dirs[i-1], this_pos[1]+xy_dirs[i], this_pos[2]+z)
                if new_pos[0] < 0 or new_pos[0] >= self.terrain_map.width \
                        or new_pos[1] < 0 or new_pos[1] >= self.terrain_map.height:
                    continue
                if new_pos[2] < self.terrain_map.data[new_pos[1]][new_pos[0]] + height_mini:
                    continue
                pos_list.append(new_pos)

        if this_pos[2] + z_dirs[0] > self.terrain_map.data[this_pos[1]][this_pos[0]] + height_mini:  # 原地下降
            new_pos = (this_pos[0], this_pos[1], this_pos[2] + z_dirs[0])
            pos_list.append(new_pos)
        if this_pos[2] + z_dirs[2] < self.terrain_map.data[this_pos[1]][this_pos[0]]+above:  # 原地上升
            new_pos = (this_pos[0], this_pos[1], this_pos[2] + z_dirs[2])
            pos_list.append(new_pos)

        return pos_list

    def cal_cost(self, p1, p2):
        """计算两点距离
        :param p1: cur node pose
        :param p2: new node pose
        :return: dis cost between p1 and p2
        """
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

    def cal_pred(self, p1, p2):
        """计算启发函数距离
        :param p1: new node pose
        :param p2: end node pose
        :return: predict cost by Heuristic functions
        """
        # 使用汉明距离作为启发函数
        # return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])

        # 使用欧式距离作为启发函数
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

    def search(self, start_p, end_p):
        """搜索过程
        :param start_p: 搜索起始点
        :param end_p: 搜索终止点
        :return:
        """
        print("start search with start point {} and end point {}".format(start_p, end_p))
        a = input("input something to go on....")
        start_time = time.time()
        open_set = PriorQueue()
        close_set = np.zeros((self.terrain_map.height, self.terrain_map.width, 1000), dtype=bool)

        start_node = Node(start_p[0], start_p[1], start_p[2])
        open_set.push(start_node)
        search_index = 0

        while not open_set.is_empty():
            # 1. get the mini-cost node from open list
            search_index += 1
            cur_node = open_set.top()
            open_set.pop()
            print("search with cur node {} with terrain elev {} and iter {}.......".format(
                cur_node.get_pos(),
                self.terrain_map.data[cur_node.get_pos()[1]][cur_node.get_pos()[0]],
                search_index))

            # if the cur node is the goal, search over
            if (np.array(cur_node.get_pos()) == end_p).all():
                end_time = time.time()
                print("SEARCH OVER!!!")
                path_iterator = PathIterator(cur_node)
                this_path = []
                for pos in path_iterator:
                    this_path.append(pos)
                this_path.reverse()
                self.path += this_path
                print("==========================================")
                print("===this path have {} node".format(len(this_path)))
                print("===total time: {}s ".format(end_time-start_time))
                print("===total length: {}".format(cur_node.get_cost_so_far()))
                break

            # 2. search and check neighborhood node
            pos_list = self.search_nei_node(cur_node, 50, 20)

            print(" nei nodes size: {}".format(len(pos_list)))

            # 3. check each pose in pos_list
            for new_pose in pos_list:
                # print("check new pose: {}".format(new_pose))
                # 3.1 check if in close set
                if close_set[new_pose[0], new_pose[1], new_pose[2]]:
                    # print("in close set...")
                    continue
                # 3.2 check if in open set
                old_node_in_open_set = open_set.in_list(new_pose)
                if old_node_in_open_set is not None:  # in open set
                    # cur_node -> new_node  -- new cost
                    new_cost_so_far = self.cal_cost(cur_node.get_pos(), new_pose) + cur_node.get_cost_so_far()
                    old_cot_so_far = old_node_in_open_set.get_cost_so_far()
                    cost_to_go = self.cal_pred(new_pose, end_p)
                    if new_cost_so_far < old_cot_so_far:
                        # print("update open set!!!")
                        open_set.del_node(new_pose)
                        new_node = Node(new_pose[0], new_pose[1], new_pose[2], father=cur_node)
                        new_node.cost_so_far = new_cost_so_far
                        new_node.cost_to_go = cost_to_go
                        # new_node.print_node()
                        open_set.push(new_node)
                else:  # not in open set
                    # print("not in open set")
                    new_cost_so_far = self.cal_cost(cur_node.get_pos(), new_pose) + cur_node.get_cost_so_far()
                    cost_to_go = self.cal_pred(new_pose, end_p)
                    new_node = Node(new_pose[0], new_pose[1], new_pose[2], father=cur_node)
                    new_node.cost_so_far = new_cost_so_far
                    new_node.cost_to_go = cost_to_go
                    # new_node.print_node()
                    open_set.push(new_node)

            # open_set.print_queue()
            print("open set size: {}".format(len(open_set.list)))

            close_set[cur_node.x, cur_node.y, cur_node.z] = True
            # if search_index % 10000 == 0:
            #     a = input("input something to go on....")


if __name__ == '__main__':

    a_star_method = Astar(waypoint_dir="./waypoint.txt", map_dir="./terrain.asc")