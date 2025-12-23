cmake_minimum_required(VERSION 3.10)
project(SLIC_Segmentation_Algorithm)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(SLIC_Segmentation_Algorithm SLIC_Segmentation_Sequential_SoA.cpp
        SLIC_Segmentation_Sequential_AoS.cpp)

target_link_libraries(SLIC_Segmentation_Algorithm ${OpenCV_LIBS})
