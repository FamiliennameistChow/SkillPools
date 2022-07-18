cmake_minimum_required(VERSION 2.8.3)
project(osqp_demo)

# Find OSQP library and headers
# find_package(osqp REQUIRED)

find_library(OSQP_LIBRARY libosqp.a HINTS osqp/build/out)
INCLUDE_DIRECTORIES(include)


add_executable(osqp_demo osqp_demo.cpp)
target_include_directories(osqp_demo PUBLIC osqp/include)
target_link_libraries(osqp_demo ${OSQP_LIBRARY} dl)


add_executable(test_qpproblem test_qpproblem.cpp)
target_include_directories(test_qpproblem PUBLIC osqp/include)
target_link_libraries(test_qpproblem ${OSQP_LIBRARY} dl)

# or...

# Link the OSQP static library
# target_link_libraries(osqp_demo PRIVATE osqp::osqpstatic)