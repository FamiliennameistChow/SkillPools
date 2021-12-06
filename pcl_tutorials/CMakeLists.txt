cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pcd_read)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/opt/homebrew/Cellar/qt@5/5.15.2_1/lib/cmake/Qt5")

find_package(PCL REQUIRED COMPONENTS common io visualization features kdtree)

find_package(Eigen3 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS}
                    include
                    ${EIGEN3_INCLUDE_DIRS}
                    )

link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})


add_executable (pcl_io_demo pcl_io_demo.cpp)
target_link_libraries (pcl_io_demo  ${PCL_LIBRARIES} )

add_executable (kdtree_search_demo kdtree_search_demo.cpp)
target_link_libraries (kdtree_search_demo ${PCL_LIBRARIES} )

add_executable(box_filter_demo box_filter_demo.cpp)
target_link_libraries(box_filter_demo ${PCL_LIBRARIES})
