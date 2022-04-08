
/**************************************************************************
 * eigen_trans_demo.cpp
 * 
 * @Author： bornchow
 * @Date: 2022.04.04
 * 
 * @Description:
 *  本程序是坐标转换示例
 *  设坐标系1 绕z旋转90度(顺时针)，然后平移(1, 0, 0) 得到坐标系2， 那么坐标1中一点P1(1, 0, 0)，在坐标系2 中的坐标P2为:
 ***************************************************************************/

#include <iostream>
# include<Eigen/Eigen>
# include<Eigen/Core>

using namespace std;

int main(){
    //由题可以 R_12 和 t_12
    // P1 = R_12 * P2 + t_12
    // P2 = R_12^-1 * (P1 - t_12 )
    Eigen::AngleAxisd rotation_vector (M_PI/2, Eigen::Vector3d(0, 0, 1));
    Eigen::Matrix3d rotation_matrix = rotation_vector.toRotationMatrix();

    Eigen::Vector3d P1 (1, 0, 0);
    Eigen::Vector3d t_12 (1, 0, 0);
    Eigen::Vector3d P2_ = P1 - t_12;
    Eigen::Vector3d P2 = rotation_matrix.inverse() * P2_;

    std::cout << " P2 : " << P2.transpose() << std::endl;

    // P2 {0, 0, 0};
    // P1 = P1 = R_12 * P2 + t_12
    Eigen::Vector3d P1_test = rotation_matrix * P2 + t_12;
    std::cout << " P1 test: " << P1_test.transpose() << std::endl;

    // T_12 2到1的转换
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();  // 虽然称为3d，实质上是4＊4的矩阵;;
    T.rotate(rotation_vector);
    T.pretranslate(t_12);
    Eigen::Vector3d P1_valid = T * P2;
    std::cout << " P1 valid: " << P1_valid.transpose() << std::endl;

    // 如果P2 为 （1， 0， 0）; 那么P1为？
    // P1 = T_12 * P2
    Eigen::Vector3d P2_other (1, 0, 0);
    Eigen::Vector3d P1_other = T * P2_other;
    std::cout << " P1 other: " << P1_other.transpose() << std::endl;

    //好了现在可以总结得出:Eigen中旋转向量 Eigen::AngleAxisd 是顺时针旋转
}

