/**************************************************************************
 * pcl_io.cpp
 * 
 * @Author： bornchow
 * @Date: 2021.09.26
 * 
 * @Description:
 * pcl库读取数据
 * 1.读入pcd文件
 * 2.读入ply文件
 * 
 ***************************************************************************/
//
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h> //读写相关函数类
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <vector>

typedef pcl::PointXYZI PointType;
using namespace std;

#define PCD_FILE
//#define PLY_FILE

int main(){

    #ifdef PCD_FILE
    string pcd_file = "../data/trj_dataset.pcd";

    pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
    if (pcl::io::loadPCDFile<PointType> (pcd_file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file pcd.pcd \n");
        return (-1);
    }
    std::cout << cloud->points.size() << std::endl;
    #endif

    #ifdef PLY_FILE
    string ply_file = "/Users/zhoubo/Downloads/bunny/reconstruction/bun_zipper.ply";

    pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);

    if(pcl::io::loadPLYFile<PointType>(ply_file, *cloud) == -1){
        PCL_ERROR ("Couldn't read file scene.pcd \n");
        return (-1);
    }
    std::cout << cloud->points.size() << std::endl;
    #endif

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("3D Viewer"));

    viewer->setBackgroundColor(1, 1, 1);
    viewer->addCoordinateSystem(1.0);

    pcl::visualization::PointCloudColorHandlerCustom<PointType> single_color(cloud, 0, 255, 0);

    viewer->addPointCloud<PointType>(cloud, single_color, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");

    while (!viewer->wasStopped()){
        viewer->spinOnce();
    }
}
