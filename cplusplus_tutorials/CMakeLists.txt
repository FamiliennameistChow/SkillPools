cmake_minimum_required(VERSION 3.20)
project(cplusplus_tutorials)

set(CMAKE_CXX_STANDARD 11)

add_executable(read_txt read_txt.cpp)

add_executable(vector_demo vector_demo.cpp)

add_executable(struct_demo struct_demo.cpp)

add_executable(struct_demo2 struct_demo2.cpp)

add_executable(struct_demo3 struct_demo3.cpp)

add_executable(queue_demo queue_demo.cpp)

add_executable(deque_demo deque_demo.cpp)

add_executable(map_demo map_demo.cpp)

add_executable(map_demo2 map_demo2.cpp)

add_executable(map_demo3 map_demo3.cpp)

add_executable(set_demo set_demo.cpp)

add_executable(double_pointer_demo double_pointer_demo.cpp)

add_executable(string_demo string_demo.cpp)

add_executable(link_demo link_demo.cpp)

add_executable(thread_demo thread_demo.cpp)

find_package(yaml-cpp CONFIG REQUIRED)
add_executable(yaml_demo yaml_demo.cpp)
target_link_libraries(yaml_demo yaml-cpp)

add_executable(stack_demo stack_demo.cpp)

add_executable(shard_ptr_demo ptr/shard_ptr_demo.cpp)

add_executable(test test.cpp)

add_executable(work1 work1.cpp)

add_executable(largeNumSub leetcode/largeNumSub.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(star Astar.cpp)
target_link_libraries(star ${OpenCV_LIBS})


