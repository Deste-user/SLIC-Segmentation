cmake_minimum_required(VERSION 3.10)
project(SLIC_Segmentation_Algorithm)

set(CMAKE_CXX_STANDARD 20)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

add_library(slic_common src/SLIC_common.cpp)
target_link_libraries(slic_common ${OpenCV_LIBS})

# Eseguibile per Structure of Arrays
add_executable(SLIC_SoA src/SLIC_Segmentation_Sequential_SoA.cpp)
target_link_libraries(SLIC_SoA slic_common ${OpenCV_LIBS})

# Eseguibile per Array of Structures
add_executable(SLIC_AoS src/SLIC_Segmentation_Sequential_AoS.cpp)
target_link_libraries(SLIC_AoS slic_common ${OpenCV_LIBS})

