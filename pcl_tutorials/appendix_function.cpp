
/**************************************************************************
 * appendix_function.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.01.24
 * 
 * @Description:
 *  介绍一些常用的函数：https://github.com/MNewBie/PCL-Notes/blob/master/appendix2_1.md
 *  01  计算程序运行时间的函数
 *  02  pcl::PointCloud::Ptr和pcl::PointCloud的两个类相互转换
 ***************************************************************************/
#include <iostream> // c++ 基础类
#include <pcl/console/time.h> //计算运行时间
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace std;
int main(){

    // --------- 01 计算程序运行时间
    pcl::console::TicToc time;
    time.tic();

    int i = 100;
    while(i>0){
        i--;
    }

    std::cout << time.toc()<< " s " << std::endl;

    // -------- 02 pcl::PointCloud::Ptr和pcl::PointCloud的两个类相互转换
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr (new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI> cloud;
    cloud = *cloud_ptr;
    cloud_ptr = cloud.makeShared();


    return 0;

}
