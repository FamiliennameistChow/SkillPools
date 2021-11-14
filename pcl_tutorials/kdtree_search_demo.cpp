
/**************************************************************************
 * kdtree_search_demo.cpp
 * 
 * @Author： bornchow
 * @Date: 2021.11.04
 * 
 * @Description:
 * 本程序演示如何在pcl中使用kdtree搜索
 * https://blog.csdn.net/weixin_46098577/article/details/119973606
 * https://blog.csdn.net/zhan_zhan1/article/details/103927896
 *
 * #include <pcl/kdtree/kdtree_flann.h>
 * pcl::KdTreeFLANN<PointType> kdtree;
 ***************************************************************************/
//

#include <iostream>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>

typedef pcl::PointXYZI  PointType;
using namespace std;

int main(){
    std::cout << "this a demo of kdtree search in PCL" << std::endl;

    string pcd_file = "../data/trj_dataset.pcd";

    pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
    if (pcl::io::loadPCDFile<PointType> (pcd_file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file scene.pcd \n");
        return (-1);
    }

    std::cout <<"point size: " << cloud->points.size() << std::endl;

    // 创建一个kdtree
    pcl::KdTreeFLANN<PointType> kdtree;
    kdtree.setInputCloud(cloud);

    PointType searchPoint;
    searchPoint.x = 1.0;
    searchPoint.y = 1.0;
    searchPoint.z = 1.0;

    // ----------------------  k临近搜索: 按给定值搜索 -------------------------
    int k = 10;

    std::vector<int> pointIdxNKNSearch; // k临近搜索到点的索引
    std::vector<float> pointNKNSquaredDistance; //k临近搜索到点到searchPoint的距离

    std::cout << "search point at (" << searchPoint.x << " " << searchPoint.y << " " <<searchPoint.z <<
    ") with K=  " << k << std::endl;

    if (kdtree.nearestKSearch(searchPoint, k, pointIdxNKNSearch, pointNKNSquaredDistance) > 0){
        for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i) {
            std::cout << "   " << cloud->points[pointIdxNKNSearch[i] ].x
            << " " << cloud->points[pointIdxNKNSearch[i] ].y
            << " " << cloud->points[pointIdxNKNSearch[i] ].z
            << " squared dis: " << pointNKNSquaredDistance[i] << " )" << std::endl;
        }
    }

    // ----------------------  k临近搜索: 按范围搜索 -------------------------

    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    float radius = 2;

    std::cout << "Neighbors within radius search at (" << searchPoint.x
              << " " << searchPoint.y
              << " " << searchPoint.z
              << ") with radius=" << radius << std::endl;


    if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
    {
        for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
            std::cout << "    "  <<   cloud->points[ pointIdxRadiusSearch[i] ].x
                      << " " << cloud->points[ pointIdxRadiusSearch[i] ].y
                      << " " << cloud->points[ pointIdxRadiusSearch[i] ].z
                      << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
    }

    return 0;
}