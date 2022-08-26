cmake_minimum_required(VERSION 3.0.2)
project(dynamic_orb_slam2)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3  -Wall  -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall   -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.4.5 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  yolact_ros
  cv_bridge
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(LIBS 
${OpenCV_LIBS} 
${EIGEN3_LIBS}
)

add_library(${PROJECT_NAME} SHARED
src/src/System.cc
src/src/Tracking.cc
src/src/LocalMapping.cc
src/src/LoopClosing.cc
src/src/ORBextractor.cc
src/src/ORBmatcher.cc
src/src/FrameDrawer.cc
src/src/Converter.cc
src/src/MapPoint.cc
src/src/KeyFrame.cc
src/src/Map.cc
src/src/MapDrawer.cc
src/src/Optimizer.cc
src/src/PnPsolver.cc
src/src/Frame.cc
src/src/KeyFrameDatabase.cc
src/src/Sim3Solver.cc
src/src/Initializer.cc
src/src/Viewer.cc
src/src/DynamicExtractor.cc)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES dynamic_orb_slam2
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${Pangolin_INCLUDE_DIRS}
)

#############
## Install ##
#############



add_executable(test_client src/test_client.cpp)
target_link_libraries(test_client ${catkin_LIBRARIES} ${LIBS})
add_dependencies(test_client yolact_ros_generate_messages_cpp)


####################
## Build examples ##
####################
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/src/Examples/RGB-D)


add_executable(rgbd_tum src/Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${catkin_LIBRARIES} ${LIBS} ${PROJECT_NAME})
add_dependencies(test_client yolact_ros_generate_messages_cpp)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/src/Examples/Stereo)

# add_executable(stereo_kitti
# src/Examples/Stereo/stereo_kitti.cc)
# target_link_libraries(stereo_kitti ${PROJECT_NAME})

# add_executable(stereo_euroc
# src/Examples/Stereo/stereo_euroc.cc)
# target_link_libraries(stereo_euroc ${PROJECT_NAME})


# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/src/Examples/Monocular)

# add_executable(mono_tum
# src/Examples/Monocular/mono_tum.cc)
# target_link_libraries(mono_tum ${PROJECT_NAME})

# add_executable(mono_kitti
# src/Examples/Monocular/mono_kitti.cc)
# target_link_libraries(mono_kitti ${PROJECT_NAME})

# add_executable(mono_euroc
# src/Examples/Monocular/mono_euroc.cc)
# target_link_libraries(mono_euroc ${PROJECT_NAME})

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/src/Examples/MaskRCNN)

# add_executable(mask_rcnn src/Examples/MaskRCNN/mask_rcnn.cc)
# target_link_libraries(mask_rcnn ${PROJECT_NAME})


# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/src/Examples/DynamicExtractor)
# include_directories(DynamicExtractor)
# add_executable(test src/Examples/DynamicExtractor/test.cc)
# target_link_libraries(test ${PROJECT_NAME})


# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/src/Examples/OpticalFlow)
# add_executable(optflow src/Examples/OpticalFlow/fback.cc)
# target_link_libraries(optflow ${PROJECT_NAME})