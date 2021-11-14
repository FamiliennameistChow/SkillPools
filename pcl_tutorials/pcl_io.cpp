/**************************************************************************
 * pcl_io.cpp
 * 
 * @Author： bornchow
 * @Date: 2021.09.26
 * 
 * @Description:
 *  
 * 
 ***************************************************************************/
//
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/random_sample.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <vector>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
typedef pcl::PointXYZI  PointType;
using namespace std;


int main(){

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("3D Viewer"));

    string pcd_file = "../data/trj_dataset.pcd";
//    string pcd_file = "/home/bornchow/workFile/test/pcl_test/pcd/trj_dataset_remove_overlap_useful.pcd";
//    string pcd_file = "/home/bornchow/workFile/test/pcl_test/pcd/sample_trj_point.pcd";
    pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);
    if (pcl::io::loadPCDFile<PointType> (pcd_file, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file scene.pcd \n");
        return (-1);
    }
    std::cout << cloud->points.size() << std::endl;
    viewer->setBackgroundColor(1, 1, 1);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> fildColor(cloud, "intensity"); // 按照 intensity 强度字段进行渲染
    viewer->addPointCloud<pcl::PointXYZI>(cloud, fildColor, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> single_color(cloud, 0, 255, 0);

    //viewer->addCoordinateSystem(1.0);

    while (!viewer->wasStopped()){
        viewer->spinOnce();
    }
//
//    return 0;
}
