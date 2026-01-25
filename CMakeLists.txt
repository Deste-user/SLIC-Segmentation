cmake_minimum_required(VERSION 3.10)
project(SLIC_Segmentation_Algorithm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-sanitize=address")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-sanitize=address")


# --- CONFIGURAZIONE MAC OS (Apple Silicon) ---
if(APPLE)
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
# Contiene TUTTE le implementazioni SLIC
add_library(slic_common
        src/SLIC_common.cpp
        src/SLIC_Algorithm_AoS.cpp
        src/SLIC_Algorithm_SoA.cpp
        src/SLIC_Algorithm_AoS_Sequential.cpp
        src/SLIC_Algorithm_AoS_Parallel.cpp
        src/SLIC_Algorithm_SoA_Sequential.cpp
        src/SLIC_Algorithm_SoA_Parallel.cpp
)

target_link_libraries(slic_common PUBLIC
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
)

# --- ESEGUIBILE UNICO PER BENCHMARK ---
add_executable(SLIC_Benchmark benchmark.cpp)
target_link_libraries(SLIC_Benchmark slic_common)
