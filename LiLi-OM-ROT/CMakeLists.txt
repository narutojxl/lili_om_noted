cmake_minimum_required(VERSION 2.8.3)
project(lili_om_rot)

## Compile as C++14, supported in ROS Kinetic and newer

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14  -g   -Wall  -O0   ")


# For Debug
# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")


## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  message_generation
  pcl_ros
  pcl_conversions
)


# find_package(Eigen3 REQUIRED) #default
list(APPEND CMAKE_PREFIX_PATH "/home/jxl/third_softwares/eigen_3_3_7/INSTALL_DIR/share/eigen3/cmake")
#set(CMAKE_PREFIX_PATH "/home/jxl/third_softwares/eigen_3_3_7/INSTALL_DIR/share/eigen3/cmake")
find_package(Eigen3 3.3.7 REQUIRED)


# find_package(Ceres REQUIRED) #default
find_package(Ceres REQUIRED PATHS "/home/jxl/third_softwares/ceres-solver-2.0.0/INSTALL/")



find_package(GTSAM REQUIRED)

message(STATUS "Using catkin version ${catkin_VERSION}")
message(STATUS "Using Ceres version ${CERES_VERSION}")

catkin_package(
    # LIBRARIES ${PROJECT_NAME}
    INCLUDE_DIRS include
    CATKIN_DEPENDS roscpp message_runtime std_msgs nav_msgs geometry_msgs
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${GTSAM_INCLUDE_DIR}
)

link_directories(
    include
    ${GTSAM_LIBRARY_DIRS}
)

## Declare a C++ executable

add_executable(Preprocessing src/Preprocessing.cpp)
target_link_libraries(Preprocessing ${catkin_LIBRARIES})

add_executable(LidarOdometry src/LidarOdometry.cpp)
target_link_libraries(LidarOdometry ${catkin_LIBRARIES} ${CERES_LIBRARIES})

add_executable(BackendFusion src/BackendFusion.cpp src/MarginalizationFactor.cpp)
target_link_libraries(BackendFusion ${catkin_LIBRARIES} ${CERES_LIBRARIES} gtsam)


