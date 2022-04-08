/**************************************************************************
 * eigen_geometry.cpp
 * 
 * @Author： bornchow
 * @Date: 2021.09.25
 * 
 * @Description:
 *  本程序演示了 Eigen 几何模块的使用方法
 *  Eigen/Geometry 模块提供了各种旋转和平移的表示
 *  https://blog.csdn.net/weixin_40353209/article/details/81356034?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1.no_search_link&spm=1001.2101.3001.4242
 ***************************************************************************/

#include <iostream>
#include <cmath>
#include <Eigen/Core>
// Eigen 几何模块
#include <Eigen/Geometry>

using namespace std;

int main(){

    // *** 3D 旋转矩阵使用  Eigen::Matrix3d  或 Eigen::Matrix3f
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity(); //单位矩阵

    // *** 旋转向量使用 AngleAxis, 它底层不直接是Matrix，但运算可以当作矩阵（因为重载了运算符）
    Eigen::AngleAxisd rotation_vector ( M_PI/2, Eigen::Vector3d ( 0,0,1 ) ); // 绕z轴旋转 M_PI/4 弧度
    Eigen::AngleAxisd rotation_vector_2 (M_PI/2, Eigen::Vector3d::UnitZ());
    cout<<"rotation matrix =\n"<<rotation_vector.matrix() <<endl;

    // ------ 旋转向量 --> 旋转矩阵--------
    rotation_matrix = rotation_vector.toRotationMatrix();
    cout << "rotation vector --> rotation_matrix \n" << rotation_matrix << endl;

    // ----- 使用旋转向量 旋转一个向量 v
    Eigen::Vector3d v (1, 0, 0);
    Eigen::Vector3d v_rotated = rotation_vector * v;
    cout<<"(1,0,0) after rotation use rotation vector= \n"<<v_rotated.transpose()<<endl;

    // ----- 使用旋转矩阵 旋转一个向量 v
    v_rotated = rotation_matrix * v;  // 左乘
    cout<<"(1,0,0) after rotation use rotation matrix = \n"<<v_rotated.transpose()<<endl;

    // *** 欧拉角 Eigen::Vector3d euler_angles表示
    // 可以将旋转矩阵直接转换成欧拉角
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles( 2, 1, 0); // ZYX顺序，即roll pitch yaw顺序
    cout << "yaw pitch roll = \n" << euler_angles.transpose() << endl;

    // **** 欧氏变换矩阵使用 Eigen::Isometry
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();  // 虽然称为3d，实质上是4＊4的矩阵
    T.rotate( rotation_vector);  // 使用rotation_vector进行旋转
    // T.rotate( rotation_matrix);  // 使用rotation_matrix进行旋转
    T.pretranslate(Eigen::Vector3d(1, 0, 0)); //把平移向量设成(1,3,4)

    // ------- 使用变换矩阵进行坐标变换
    Eigen::Vector3d v_transformed = T*v; // 左乘 相当于 R*v + t
    cout << " after transformed: \n" << v_transformed.transpose() << endl;

    // **** 四元数
    // --- 初始化Quaterniond
    Eigen::Quaterniond q (rotation_vector); // 第一种方式 使用旋转向量
    Eigen::Quaterniond q1 (rotation_matrix); // 第二种方式 使用旋转矩阵
    double w, x, y, z;                       // 第三种方式  注意w是实部 在前 !!!!!
    Eigen::Quaterniond q2(w, x, y, z);   //Quaternion (const Scalar &w, const Scalar &x, const Scalar &y, const Scalar &z)
    Eigen::Quaterniond q3(Eigen::Vector4d(x, y, z, w)); // 第四种方式  注意w是实部 在后 !!!!!
    Eigen::Quaterniond q4 = Eigen::Quaterniond ( rotation_vector );

    //在Quaternion内部的保存中，虚部在前，实部在后
    cout<<"quaternion = \n"<<q.coeffs() <<endl;   // 请注意coeffs的顺序是(x,y,z,w),w为实部，前三者为虚部

    // ---- 旋转矩阵 --> 四元数
    Eigen::Quaterniond q5 (rotation_matrix);

    Eigen::Quaterniond q6;
    q6 = rotation_matrix;

    // ----- 四元数获取值
    cout << q6.x() << endl;
    cout << q6.y() << endl;
    cout << q6.z() << endl;
    cout << q6.w() << endl;

    // ------ 四元数 --> 旋转矩阵
    Eigen::Matrix3d R1 = q.toRotationMatrix();
    cout << "Quaterniond to RotationMatrix: \n" << R1 << endl;

    // ----- 使用四元数旋转向量
    Eigen::Vector3d v_rotated_1 = q*v;

    cout << "v_rotated_1 =\n" << v_rotated_1.transpose() << endl;

    return 0;
}
