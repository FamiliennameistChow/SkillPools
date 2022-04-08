
/**************************************************************************
 * pcl_io_write.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.01.24
 * 
 * @Description:
 * 将点云写入文件
 *  
 ***************************************************************************/
#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

using namespace std;
typedef pcl::PointXYZI PointType;

int main(){

    pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>());

    cloud->width = 5;
    cloud->height = 1;
    cloud->is_dense = false;
    cloud->points.resize(cloud->width*cloud->height);

    for (int i = 0; i < cloud->points.size(); ++i) {
        cloud->points[i].x = 1024*rand() / (RAND_MAX + 1.0f);
        cloud->points[i].y = 1024*rand() / (RAND_MAX + 1.0f);
        cloud->points[i].z = 1024*rand() / (RAND_MAX + 1.0f);
        cloud->points[i].intensity = 1024*rand() / (RAND_MAX + 1.0f);
    }

    pcl::io::savePCDFile("savepcd.pcd", *cloud);
    pcl::io::savePCDFileASCII("savepcd_ascii.pcd", *cloud);
    std::cout << "save pcd finished!!!" << std::endl;


    return 0;
}

