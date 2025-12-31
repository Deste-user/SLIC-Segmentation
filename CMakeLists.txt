cmake_minimum_required(VERSION 3.10)
project(SLIC_Segmentation_Algorithm)

set(CMAKE_CXX_STANDARD 20)

# --- CONFIGURAZIONE MAC OS (Apple Silicon) ---
# Questo deve stare PRIMA di chiamare find_package(OpenMP)
if(APPLE)
    # Dice a CMake di cercare anche dove Homebrew installa libomp
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
endif()

# --- RICERCA DIPENDENZE ---
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# --- INCLUDE ---
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

if (OPENMP_CXX_FOUND)
    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
endif()

# --- LIBRERIA COMUNE ---
add_library(slic_common
        src/SLIC_common.cpp
        src/Parallel_SLIC_SoA.cpp
)

# Collega OpenCV e OpenMP alla libreria comune.
# PUBLIC: chi usa 'slic_common' erediterà automaticamente queste librerie.
target_link_libraries(slic_common PUBLIC
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
)

# --- ESEGUIBILE Structure of Arrays ---
add_executable(SLIC_SoA src/SLIC_Segmentation_Sequential_SoA.cpp)
target_link_libraries(SLIC_SoA slic_common)

# --- ESEGUIBILE Array of Structures ---
add_executable(SLIC_AoS src/SLIC_Segmentation_Sequential_AoS.cpp
        src/Parallel_SLIC_AoS.cpp)
target_link_libraries(SLIC_AoS slic_common)

add_executable(
SLIC_SoA_Parallel src/Parallel_SLIC_SoA.cpp
)
target_link_libraries(SLIC_SoA_Parallel slic_common)

add_executable(
SLIC_AoS_Parallel src/Parallel_SLIC_AoS.cpp
)
target_link_libraries(SLIC_AoS_Parallel slic_common)