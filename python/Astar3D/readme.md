# 三维A星算法

## install

```
matplotlib == 3.5.0
opencv 4.5.1
```

terrain.asc 为地图数据，代码中将地图放缩至 100 * 100 * 500

waypoint.txt 指定起始点与目标点， 格式为:
```
x坐标值 y坐标值 该坐标下距离地面的相对高度
```
waypoint.txt 可以指定多个路点：
例如：
```
0 0 10
10 10 10
20 20 10
```
以上表示搜索 (0, 0, 10) -> (10, 10, 10) 和 (10, 10, 10) -> (20, 20, 10)

# how to run

```
python Astar3D.py
```

ternimal中显示
```
input something to go on....
```
键入任意值启动a*