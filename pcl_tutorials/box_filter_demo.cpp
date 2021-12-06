
/**************************************************************************
 * box_filter_demo.cpp
 * 
 * @Author： bornchow
 * @Date: 2021.11.23
 * 
 * @Description:
 *  本程序演示如何过滤立方体内或外的点云
 *  http://pointclouds.org/documentation/classpcl_1_1_crop_box_3_01pcl_1_1_p_c_l_point_cloud2_01_4.html
 *  https://blog.csdn.net/ethan_guo/article/details/80359313
 ***************************************************************************/
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/crop_box.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <string>

typedef pcl::PointXYZI  PointType;
using namespace std;

int main(){

    string pcd_file = "../data/trj_dataset.pcd";
    pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
    if (pcl::io::loadPCDFile<PointType> (pcd_file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file scene.pcd \n");
        return (-1);
    }

    std::cout << cloud->points.size() << std::endl;


    pcl::PointCloud<PointType>::Ptr filter_cloud (new pcl::PointCloud<PointType>);

    pcl::CropBox<PointType> box_filter; //需要引用 #include <pcl/filters/crop_box.h>

    float box_size = 2;
    // 设置box大小
    // setMin setMax 并不是box中任意两个对角点，一定要找到值最小与最大的两个对角点
    box_filter.setMin(Eigen::Vector4f(-box_size/2, -box_size/2, -box_size/2, 1.0));
    box_filter.setMax(Eigen::Vector4f(box_size/2, box_size/2, box_size/2, 1.0));

    // 设置box的位姿
    box_filter.setTransform();

    // 设置滤除
    box_filter.setNegative(false); //false 是表示去除box外的点，true 表示去除box内的点

    // 设置输入输出
    box_filter.setInputCloud(cloud);
    box_filter.filter(*filter_cloud);


    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(1, 1, 1);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> single_color(cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZI>(filter_cloud, single_color, "cloud");
    viewer->addCoordinateSystem(1.0);

    while (!viewer->wasStopped()){
        viewer->spinOnce();
    }

    return 0;
}

