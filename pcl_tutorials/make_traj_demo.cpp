/**************************************************************************
 * pcl_io.cpp
 *
 * @Author： bornchow
 * @Date: 2022.05.13
 *
 * @Description:
 * 制作轨迹库代码
 *
 ***************************************************************************/

#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <vector>
#include <pcl/kdtree/kdtree_flann.h>
#include <fstream>

using namespace std;
typedef pcl::PointXYZI PointType;

int main(){

    //param
    vector<float> vel_set {0, 1, 0.1};
    vector<float> ang_set {-1, 1, 0.1};
    std::map<int, std::vector<float>> action_table;

    //
    std::cout << " ======== param ======== " << std::endl;
    std::cout << "vel set | " << vel_set[0] << " " << vel_set[1] << " " << vel_set[2] << std::endl;
    std::cout << "ang set | " << ang_set[0] << " " << ang_set[1] << " " << ang_set[2] << std::endl;

    int grounp_nums = (ang_set[1] - ang_set[0])/ ang_set[2] + 1;
    std::cout << grounp_nums << std::endl;

    pcl::PointCloud<PointType>::Ptr traj_ds  (new pcl::PointCloud<PointType>());
    std::vector<pcl::PointCloud<PointType>::Ptr> v_traj_ds;
    float t = 0;
    float ind = 0.0;
    for(float vel = vel_set[0]; vel <= vel_set[1]+0.1; ){
        if(vel == 0){
            vel += vel_set[2];
            continue;
        }

        for(float ang = ang_set[0]; ang <= ang_set[1]+0.1; ){
            pcl::PointCloud<PointType>::Ptr this_trj (new pcl::PointCloud<PointType>());
            while (t <= 1.0){
                PointType p;
                p.x = vel * t * cos(ang*t);
                p.y = vel * t * sin(ang*t);
                p.intensity = ind;

                if(abs(ang) < 0.5 && abs(vel) < 0.5){
                    t += 0.01;
                } else{
                    t += 0.01;
                }
                this_trj->points.push_back(p);
            }
            std::vector<float> action;
            action.push_back(vel);
            action.push_back(ang);
            action_table.insert(make_pair(ind, action));
            std::cout << "vel: " << vel << " ang: " << ang << std::endl;
            v_traj_ds.push_back(this_trj);
            ang += ang_set[2];
            t = 0;
            ind++;

        }
        vel += vel_set[2];
    }

//    std::cout << " ind: " << ind <<std::endl;
    std::cout << "total traj nums: " << v_traj_ds.size() << std::endl;
    int dataset_size = v_traj_ds.size();

//    std::cout << " ----- action table -----" << std::endl;
//    for(auto ac : action_table){
//        std::cout << " index: " << ac.first << " vel: " << ac.second[0] << " ang: " << ac.second[1] << std::endl;
//    }


//    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("3D Viewer"));
//    viewer->setBackgroundColor(1, 1, 1);
//    // viewer->addCoordinateSystem(1.0);
//
//    // pcl::visualization::PointCloudColorHandlerCustom<PointType> single_color(traj_ds, 0, 255, 0);
//    pcl::visualization::PointCloudColorHandlerGenericField<PointType> fildColor (traj_ds, "intensity");
//    // 按照 intensity 强度字段进行渲染
//
//    viewer->addPointCloud<PointType>(traj_ds, fildColor, "cloud");
//    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
//
//    while (!viewer->wasStopped()){
//        viewer->spinOnce();
//    }

    // remove overlap
    vector<int> remove_indexs;

    for(int index=0; index< v_traj_ds.size() - grounp_nums; index++){
        pcl::PointCloud<PointType>::Ptr this_traj = v_traj_ds[index];
        pcl::PointCloud<PointType>::Ptr this_dataset (new pcl::PointCloud<PointType>());
        for(int i=0; i<v_traj_ds.size();i++){
            if(i / grounp_nums == index / grounp_nums || i < index){ //  同族
                continue;
            } else{
                *this_dataset += *v_traj_ds[i];
            }
        }

        float radius = 0.05;
        pcl::KdTreeFLANN<PointType> kdtree;
        kdtree.setInputCloud(this_dataset);
        std::vector<int> this_traj_overlap_cntrs(dataset_size, 0); //记录该轨迹与其他轨迹的重叠次数
        int this_traj_size = this_traj->points.size();

        for(int a=0; a<this_traj->points.size(); a++){ // 遍历this_traj
            PointType searchPoint = this_traj->points[a];

            std::vector<int> pointIdxRadiusSearch;
            std::vector<float> pointRadiusSquaredDistance;

            if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
            {
                std::vector<int> overlap_index;
                for (size_t i = 0; i < pointIdxRadiusSearch.size (); ++i){
                    overlap_index.push_back(this_dataset->points[ pointIdxRadiusSearch[i] ].intensity);
                }

                // 去掉重复index，也就是说，一个点只能与一条轨迹重合一次
                std::sort(overlap_index.begin(), overlap_index.end());
                overlap_index.erase(unique(overlap_index.begin(),overlap_index.end()),
                                    overlap_index.end());

                for(int i=0; i<overlap_index.size();i++){
                    this_traj_overlap_cntrs[overlap_index[i]] += 1;
                }
            }

            //一个轨迹上的点处理完毕
        }
        //std::cout << " this traj cntr......" << std::endl;
        for(int i=index; i< this_traj_overlap_cntrs.size(); i++){
            float radio = this_traj_overlap_cntrs[i] * 1.0 / this_traj_size;
            std::cout << "this traj: " << index <<" total size:" << this_traj_size << " target traj: "<< i <<
            " overlap nums: " << this_traj_overlap_cntrs[i] << " radio: " << radio << std::endl;

            if( radio > 0.95){
                remove_indexs.push_back(index); // 一旦该轨迹与其他轨迹的重合度大于一定值,则该轨迹需要被remove
                break;
            }
        }

        //一条轨迹处理完毕

    }


    std::cout << " need remove traj index size: " << remove_indexs.size() << std::endl;

    pcl::PointCloud<PointType>::Ptr traj_ds_remove_overlap  (new pcl::PointCloud<PointType>());
    std::vector<int> remain_index;

    for(int i =0; i<v_traj_ds.size(); i++){
        bool needRemove = false;
        for(int j=0; j < remove_indexs.size(); j++){
            if(i == remove_indexs[j]){
                needRemove = true;
            }

            if(i % grounp_nums == 0 || i % grounp_nums == grounp_nums -1 ){ //保留每族的第一和最后条轨迹
                needRemove = false;
            }
        }
        if(!needRemove){
            *traj_ds_remove_overlap += *v_traj_ds[i];
            remain_index.push_back(i);
        }
    }

    std::cout << " remaind_size: " << remain_index.size() << std::endl;

    string fileName = "/Users/zhoubo/Documents/SkillPools/pcl_tutorials/data/action_table.txt";

    ofstream outfile(fileName);

    for(auto re_ind : remain_index){
        auto it = action_table.find(re_ind);
        if(it != action_table.end()){
            outfile << it->first << " " << it->second[0] << " " << it->second[1] << std::endl;
        }
    }

    pcl::io::savePCDFile("/Users/zhoubo/Documents/SkillPools/pcl_tutorials/data/traj_ds_remove.pcd",
                         *traj_ds_remove_overlap);


    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(1, 1, 1);
    // viewer->addCoordinateSystem(1.0);

    // pcl::visualization::PointCloudColorHandlerCustom<PointType> single_color(traj_ds_remove_overlap, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerGenericField<PointType> fildColor (traj_ds_remove_overlap,
                                                                                 "intensity");
    // 按照 intensity 强度字段进行渲染

    viewer->addPointCloud<PointType>(traj_ds_remove_overlap, fildColor, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");

    while (!viewer->wasStopped()){
        viewer->spinOnce();
    }



}