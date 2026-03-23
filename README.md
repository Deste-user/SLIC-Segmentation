# Parallel SLIC Superpixels

## Project Description
Breve introduzione al progetto. Spiega cos'è l'algoritmo SLIC (Simple Linear Iterative Clustering) e qual è l'obiettivo principale di questa repository (es. accelerare l'algoritmo sfruttando il multithreading su CPU, analizzare i colli di bottiglia della memoria e testare diverse strategie di sincronizzazione).

In this repository, there is a analisys of a segmentation algorithm using in the Computer Vision to make a segmentation preserving the boundaries of the original image. The complexity of the algorithm is $O(N)$ where N is the number of pixels. 
The principal goal of the repository is to try accellerate the algorithm using multithreading of the CPU with the OpenMP framework.
Here below, there are the original photo and the segmented photo.
![Original Image](\generated_graphs\original.png)
![Segmented Image](\generated_graphs\segmentated.png)

## Optimization Tecniques
The implementation uses a execution file benchmark to make an analysis to understand the better configuration to improve the performance. These analysis comprend:
- **Best number of threads:** understand the optimed number of threads to resolve the algorithm with the right speed up.
- **Memory Layout:** comparison between AoS and SoA.
- **Synchronization:** implementation to avoid race conditions with `Atomics` vs `Reduction`.
- **OpenMP Scheduling:** analysis of performance changing schedule policy (`static`, `dynamic`, `guided`) and chunk dimentions.


## Requirement's System
To run the code, the user must have:
* Compilator C++ with support of OpenMP (es. `g++` o `clang++`).
* OpenCV to load the images.
* Python 3.x, `pandas`, `matplotlib`, e `seaborn` to generate all the plots.

## ⚙️ Compilation
If the user want to build and execute the benchmark must use these rows:
```bash
mkdir build && cd build
cmake ..
make
```
After that the user can generate the plots  using this row
