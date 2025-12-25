cmake_minimum_required(VERSION 3.10)
project(SLIC_Segmentation_Algorithm)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Eseguibile per Structure of Arrays
add_executable(SLIC_SoA SLIC_Segmentation_Sequential_SoA.cpp)
target_link_libraries(SLIC_SoA ${OpenCV_LIBS})

# Eseguibile per Array of Structures
add_executable(SLIC_AoS SLIC_Segmentation_Sequential_AoS.cpp)
target_link_libraries(SLIC_AoS ${OpenCV_LIBS})

