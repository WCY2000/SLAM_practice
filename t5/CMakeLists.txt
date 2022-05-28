cmake_minimum_required( VERSION 2.8 )
project( Task5 )
 
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-O3" )
 
include_directories( )
find_package(Eigen3)
include_directories(
  ${EIGEN3_INCLUDE_DIR}
)

find_package(OpenCV REQUIRED) 


#add_executable( 3_1 3_1.cpp )
add_executable( 5_2 5_2.cpp )
target_link_libraries(5_2  ${OpenCV_LIBS}) 