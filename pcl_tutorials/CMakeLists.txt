cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pcd_read)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(PCL REQUIRED )

find_package(Eigen3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS}
                    include
                    ${EIGEN3_INCLUDE_DIRS}
                    )

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})


add_executable (pcl_io pcl_io.cpp)
target_link_libraries (pcl_io   ${PCL_LIBRARIES} )

add_executable (kdtree_search_demo kdtree_search_demo.cpp)
target_link_libraries (kdtree_search_demo  ${PCL_LIBRARIES} )

