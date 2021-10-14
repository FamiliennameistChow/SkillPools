/**************************************************************************
 * eigen_transfrom.cpp
 * 
 * @Author： bornchow
 * @Date: 2021.09.25
 * 
 * @Description:
 *  本程序演示 eigen 构造欧式变换矩阵
 *  https://blog.csdn.net/HelloJinYe/article/details/106926187?utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link
 ***************************************************************************/

# include <iostream>
# include<Eigen/Eigen>
# include<Eigen/Core>

using namespace std;

int main(){

    // 1. 使用Eigen::Isometry3d构造变换矩阵
    Eigen::Isometry3d T1 = Eigen::Isometry3d::Identity();  //虽然称为3d，实质上是4＊4的矩阵
    std::cout << "init T1" << std::endl;
    std::cout << T1.matrix() << std::endl;

    // 不能直接赋值
    //    T1<< 1.000000e+00, 1.197624e-11, 1.704639e-10, 3.214096e-14,
    //            1.197625e-11, 1.197625e-11, 3.562503e-10, -1.998401e-15,
    //            1.704639e-10, 3.562503e-10, 1.000000e+00, -4.041212e-14,
    //                       0,            0,            0,

    // 对各个元素进行赋值
    T1(0,0) = 1.000000e+00, T1(0,1) = 1.197624e-11, T1(0,2) = 1.704639e-10, T1(0,3) = 3.214096e-14;
    T1(1,0) = 1.197625e-11, T1(1,1) = 1.197625e-11, T1(1,2) = 3.562503e-10, T1(1,3) = -1.998401e-15;
    T1(2,0) = 1.704639e-10, T1(2,1) = 3.562503e-10, T1(2,2) = 1.000000e+00, T1(2,3) = -4.041212e-14;
    T1(3,0) =            0, T1(3,1) =            0, T1(3,2) =            0, T1(3,3) =             1;


    // 通过旋转矩阵和平移向量赋值
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    rotation_matrix << 1.0, 0.0, 0.0,
                       0.0, 1.0, 0.0,
                       0.0, 0.0, 1.0;
    Eigen::Vector3d t;
    t << 3.2, -1.9, 2.0;

    T1.rotate(rotation_matrix);
    T1.pretranslate(t);

    std::cout << "----value---" << std::endl;
    std::cout << T1.matrix() << std::endl;
    // 注意不能直接变换矩阵赋值

    // 2. Eigen::Matrix4d构造变换矩阵
    Eigen::Matrix4d T2;
    std::cout << "----init T2----" << std::endl;
    T2.setIdentity();
    std::cout << T2.matrix() << std::endl;

    // 赋值
    // (1) 直接赋值
    T2 << 1.0, 0.0, 0.0, 1.0,
          0.0, 1.0, 0.0, 1.2,
          0.0, 0.0, 1.0, 2,
          0.0, 0.0, 0.0, 1.0;
    std::cout << " ----- " << std::endl;
    std::cout << T2.matrix() << std::endl;

    // (2)通过旋转矩阵和平移向量赋值
    T2.setIdentity();
    T2.block<3, 3>(0, 0) = rotation_matrix;
    T2.topRightCorner(3, 1) = t;
    std::cout << " ----- " << std::endl;
    std::cout << T2.matrix() << std::endl;

    
    return 0;
}