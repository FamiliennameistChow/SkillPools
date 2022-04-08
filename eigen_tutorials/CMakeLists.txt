cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(eigen_samples)

find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(eigen_transfrom eigen_transfrom.cpp)

add_executable(eigen_geometry eigen_geometry.cpp)

add_executable(eigen_trans_demo eigen_trans_demo.cpp)