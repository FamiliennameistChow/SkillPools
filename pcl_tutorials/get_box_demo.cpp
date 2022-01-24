/**************************************************************************
 * get_box_demo.cpp
 * 
 * @Author： bornchow
 * @Date: 2021.12.16
 * 
 * @Description:
 *  本程序展示如何获取点云的外接矩形框
 * 
 ***************************************************************************/

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/segmentation/extract_clusters.h>

#include <iostream>
#include <string>

typedef pcl::PointXYZI  PointType;
using namespace std;

int main(){

    string pcd_file = "../data/0.pcd";
    pcl::PointCloud<PointType>::Ptr process_pc_pass (new pcl::PointCloud<PointType>);
    if (pcl::io::loadPCDFile<PointType> (pcd_file, *process_pc_pass) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file scene.pcd \n");
        return (-1);
    }
    std::cout << process_pc_pass->points.size() << std::endl;


    //欧式聚类
    std::vector<pcl::PointIndices> ece_inlier;
    pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
    tree->setInputCloud(process_pc_pass);

    pcl::EuclideanClusterExtraction<PointType> ece_;
    ece_.setClusterTolerance (0.2); //设置近邻搜索的搜索半径为10cm
    ece_.setMinClusterSize (10);//设置一个聚类需要的最少点数目为100
    ece_.setMaxClusterSize (25000); //设置一个聚类需要的最大点数目为25000
    ece_.setSearchMethod(tree);
    ece_.setInputCloud (process_pc_pass);
    ece_.extract (ece_inlier);//从点云中提取聚类，并将点云索引保存在ece_inlier中

    std::cout <<"seg num: " << ece_inlier.size() << std::endl;

    PointType thisPosePoint;
    pcl::KdTreeFLANN<PointType> kdtree2;
    int k = 1;
    std::vector<int> pointIdxNKNSearch(k);
    std::vector<float> pointNKNSquaredDistance(k);
    thisPosePoint.x = 11.0;
    thisPosePoint.y = 0.0;
    thisPosePoint.z = -0.5;

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(1, 1, 1);
    float min_dis = 1000;
    int min_index = 1000;
    float this_dis = 10;
    for (int i = 0; i < ece_inlier.size(); i++)
    {
        pcl::PointCloud<PointType>::Ptr cloud_seg (new pcl::PointCloud<PointType>());
        std::vector<int> this_inlier = ece_inlier[i].indices;
        pcl::copyPointCloud(*process_pc_pass, this_inlier, *cloud_seg);//按照索引提取点云数据

        kdtree2.setInputCloud(cloud_seg);
        if ( kdtree2.nearestKSearch (thisPosePoint, k, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 ){
            this_dis = pointNKNSquaredDistance[0];
            if(this_dis < min_dis){
                std::cout << cloud_seg->points[pointIdxNKNSearch[0]] << std::endl;
                std::cout <<"dis: " << this_dis << std::endl;
                min_dis = this_dis;
                min_index = i;
            }
        }
    }

    std::cout << "min_index: " << min_index << std::endl;

    std::vector<int> this_inlier = ece_inlier[min_index].indices;
    pcl::PointCloud<PointType>::Ptr choosed_pc (new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*process_pc_pass, this_inlier, *choosed_pc);//按照索引提取点云数据

    std::cout << choosed_pc->points.size() << std::endl;

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> single_color(choosed_pc, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZI>(choosed_pc, single_color, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");


//    pcl::MomentOfInertiaEstimation <PointType> feature_extractor;
//    feature_extractor.setInputCloud (cloud);
//    feature_extractor.compute ();
//    std::vector <float> moment_of_inertia;
//    std::vector <float> eccentricity;
//    PointType min_point_AABB;
//    PointType max_point_AABB;
//    PointType min_point_OBB;
//    PointType max_point_OBB;
//    PointType position_OBB;
//    Eigen::Matrix3f rotational_matrix_OBB;
//    float major_value, middle_value, minor_value;
//    Eigen::Vector3f major_vector, middle_vector, minor_vector;
//    Eigen::Vector3f mass_center;
//
//    feature_extractor.getMomentOfInertia (moment_of_inertia);
//    feature_extractor.getEccentricity (eccentricity);
//    feature_extractor.getAABB (min_point_AABB, max_point_AABB);
//    feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
//    feature_extractor.getEigenValues (major_value, middle_value, minor_value);
//    feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);
//    feature_extractor.getMassCenter (mass_center);
//
//    //obb外接立方体，最小外接立方体


//    Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
//    Eigen::Quaternionf quat(rotational_matrix_OBB);
//    viewer->addCube(position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB");
//    viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "OBB");
//


    while (!viewer->wasStopped()){
        viewer->spinOnce();
    }

    return 0;
}



