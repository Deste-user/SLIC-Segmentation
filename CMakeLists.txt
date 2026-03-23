#Se versione sul pc è più vecchia di 3.16, errore
cmake_minimum_required(VERSION 3.16)

# Nome del progetto e specifica del linguaggio
project(SLIC_Segmentation_Algorithm LANGUAGES CXX)

# Imposta lo standard C++20 per tutti i target
set(CMAKE_CXX_STANDARD 20)
# CMAKE_CXX_STANDARD_REQUIRED assicura che se il compilatore non supporta C++20, errore.
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dice al compilatore di generare codice nativo ottimizzato per la piattaforma su cui stiamo compilando.
set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0")

# Se WINDOWS
if(MSVC)
    # /O2 equivale a -O3 (Massima ottimizzazione)
    # /fp:fast equivale a -ffast-math
    # /arch:AVX2 abilita le istruzioni vettoriali avanzate per CPU moderne
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /fp:fast /arch:AVX2")
else()
    # Flag comuni per Linux (GCC) e macOS (Clang)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")

    if(APPLE)
        # Su macOS Apple Silicon, l'ottimizzazione specifica è -mcpu
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
    else()
        # Su Linux (architetture Intel/AMD) usiamo i flag nativi
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
    endif()
endif()

# Se siamo su macOS, OMP non è incluso di default
if(APPLE)
    # Trova Homebrew per installare OMP
    find_program(HOMEBREW_PREFIX brew)
    if(HOMEBREW_PREFIX)
        # Se Homebrew è installato, si usa per trovare il percorso di libomp
        execute_process(
                COMMAND ${HOMEBREW_PREFIX} --prefix libomp
                # Cattura l'output in una variabile CMake
                OUTPUT_VARIABLE LIBOMP_PREFIX
                # Rimuove eventuali spazi bianchi alla fine dell'output
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        # Se libomp è stato trovato, configura i flag per OpenMP
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES omp)
        set(OpenMP_CXX_LIB_NAMES omp)
        set(OpenMP_omp_LIBRARY ${LIBOMP_PREFIX}/lib/libomp.dylib)
    endif()
endif()

#Se non è MacOS, OMP e OpenCV si trovano normalmente
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)


# Si crea una libreria statica( cioè una libreria che viene incollata direttamente nell'eseguibile al momento della compilazione)
#chiamata 'slic_common' che contiene tutto il codice comune agli algoritmi SLIC.
add_library(slic_common STATIC
        src/SLIC_common.cpp
        src/SLIC_Algorithm_AoS.cpp
        src/SLIC_Algorithm_SoA.cpp
        src/SLIC_Algorithm_AoS_Sequential.cpp
        src/SLIC_Algorithm_AoS_Parallel.cpp
        src/SLIC_Algorithm_SoA_Sequential.cpp
        src/SLIC_Algorithm_SoA_Parallel.cpp
)

# PUBLIC significa che chiunque usi 'slic_common' erediterà anche questi include e librerie
# Dice alla tua libreria dove trovare i file di OpenCV.
target_include_directories(slic_common PUBLIC include ${OpenCV_INCLUDE_DIRS})
#Linka fisicamente il codice compilato di OpenCV e OpenMP alla tua libreria.
target_link_libraries(slic_common PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

#Eseguibile che esegue i benchmark
add_executable(SLIC_Benchmark benchmark.cpp)


# L'eseguibile viene linkato allo slic_common. Eredita tutto ciò che è pubblico in slic_common.
target_link_libraries(SLIC_Benchmark PRIVATE slic_common)