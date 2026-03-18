cmake_minimum_required(VERSION 3.10)

project(SLIC_Segmentation_Algorithm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0")

# Configura OpenMP per Homebrew su macOS
if(APPLE)
    find_program(HOMEBREW_PREFIX brew)
    if(HOMEBREW_PREFIX)
        execute_process(
            COMMAND ${HOMEBREW_PREFIX} --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES omp)
        set(OpenMP_CXX_LIB_NAMES omp)
        set(OpenMP_omp_LIBRARY ${LIBOMP_PREFIX}/lib/libomp.dylib)
    endif()
endif()

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# --- LIBRERIA COMUNE ---
add_library(slic_common
        src/SLIC_common.cpp
        src/SLIC_Algorithm_AoS.cpp
        src/SLIC_Algorithm_SoA.cpp
        src/SLIC_Algorithm_AoS_Sequential.cpp
        src/SLIC_Algorithm_AoS_Parallel.cpp
        src/SLIC_Algorithm_SoA_Sequential.cpp
        src/SLIC_Algorithm_SoA_Parallel.cpp
)

target_include_directories(slic_common PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(slic_common PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# --- ESEGUIBILE BENCHMARK ---
add_executable(SLIC_Benchmark benchmark.cpp)

target_include_directories(SLIC_Benchmark PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(SLIC_Benchmark PUBLIC slic_common ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

