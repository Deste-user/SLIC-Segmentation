// In this file we implement the parallel version of SLIC using Structure of Arrays (SoA) approach.
#include "SLIC_common.h"
#include <opencv2/opencv.hpp>

#include <omp.h>
#include "SLIC_Algorithm_SoA_Parallel.h"




// Use the pixel centric version - each pixel checks all superpixels in its 2Sx2S regio
void SLIC_Algorithm_SoA_Parallel::iteration() {
    const int rows = this->image_lab.rows;
    const int cols = this->image_lab.cols;
    const int grid_w = cols/S;
    const int grid_h = rows/S;

#pragma omp parallel default(none) shared(rows, cols, grid_w, grid_h)
    {
        if(!this->use_tiling) {
            #pragma omp for schedule(runtime)
            for (int y = 0; y < rows; y++) {
                int grid_y = y / S;
                for (int x=0; x < cols; x++){
                    int idx = x + cols * y;
                    int grid_x = x / S;

                    float val_L = img->L[idx];
                    float val_a = img->A[idx];
                    float val_b = img->B[idx];
                    int pos_x = img->x[idx];
                    int pos_y = img->y[idx];
                    double min_distance = DBL_MAX;
                    int best_k = -1;
                    bool changed = false;

                    for (int ny = -1; ny <= 1; ny++) {
                        int ky = grid_y + ny;
                        if (ky < 0 || ky >= grid_h) continue;
                        int k_row_offset = ky * grid_w;
                        for (int nx = -1; nx <= 1; nx++) {
                            int kx = grid_x + nx;
                            if (kx < 0 || kx >= grid_w) continue;
                            int k = k_row_offset + kx;
                            if (abs(super_pixels->centroid_x[k] - x) < S &&
                                abs(super_pixels->centroid_y[k] - y) < S) {
                                double d = distance_SLIC(super_pixels->val_L[k], super_pixels->val_a[k],
                                                         super_pixels->val_b[k], super_pixels->centroid_x[k],
                                                         super_pixels->centroid_y[k],
                                                         val_L, val_a, val_b, pos_x, pos_y, S, m);
                                if (d < min_distance) {
                                    min_distance = d;
                                    best_k = k;
                                }
                            }
                        }
                    }
                    if (best_k >= 0) {
                        img->distances[idx] = (float) min_distance;
                        img->labels[idx] = best_k;
                    }
                }
            }
        }else{
            int current_tile_size = 2 * this->S;
            #pragma omp for collapse(2) schedule(static)
            for (int by = 0; by < rows; by += current_tile_size) {
                for (int bx = 0; bx < cols; bx += current_tile_size) {
                    int y_end = std::min(by + current_tile_size, this->image_lab.rows);
                    int x_end = std::min(bx + current_tile_size, this->image_lab.cols);
                    for (int y = by; y < y_end; y++) {
                        int grid_y = y / S;
                        for (int x = bx; x < x_end; x++) {
                            int idx = x + cols * y;
                            int grid_x = x / S;

                            float val_L = img->L[idx];
                            float val_a = img->A[idx];
                            float val_b = img->B[idx];
                            int pos_x = img->x[idx];
                            int pos_y = img->y[idx];
                            double min_distance = DBL_MAX;
                            int best_k = -1;

                            for (int ny = -1; ny <= 1; ny++) {
                                int ky = grid_y + ny;
                                if (ky < 0 || ky >= grid_h) continue;
                                int k_row_offset = ky * grid_w;
                                for (int nx = -1; nx <= 1; nx++) {
                                    int kx = grid_x + nx;
                                    if (kx < 0 || kx >= grid_w) continue;
                                    int k = k_row_offset + kx;
                                    if (abs(super_pixels->centroid_x[k] - x) < S &&
                                        abs(super_pixels->centroid_y[k] - y) < S) {
                                        double d = distance_SLIC(super_pixels->val_L[k], super_pixels->val_a[k],
                                                                 super_pixels->val_b[k], super_pixels->centroid_x[k],
                                                                 super_pixels->centroid_y[k],
                                                                 val_L, val_a, val_b, pos_x, pos_y, S, m);
                                        if (d < min_distance) {
                                            min_distance = d;
                                            best_k = k;
                                        }
                                    }
                                }
                            }
                            if (best_k >= 0) {
                                img->distances[idx] = (float) min_distance;
                                img->labels[idx] = best_k;
                            }
                        }
                    }
                }
            }
        }
    }
}

void SLIC_Algorithm_SoA_Parallel::update_centroids() {
    // Parallelizzare l'inizializzazione per first-touch NUMA-friendly
#pragma omp parallel for schedule(static)
    for (int k = 0; k < K_max; k++) {
        buff_x[k] = 0.0;
        buff_y[k] = 0.0;
        buff_L[k] = 0.0;
        buff_a[k] = 0.0;
        buff_b[k] = 0.0;
        buff_count[k] = 0;
    }

    if (this->reduction_parallel) {
#pragma omp parallel
        {
#pragma omp for schedule(runtime) \
reduction(+: buff_x[:K_max], buff_y[:K_max], buff_L[:K_max], \
buff_a[:K_max], buff_b[:K_max], buff_count[:K_max])
            for (int idx = 0; idx < N; idx++) {
                int lbl = img->labels[idx];
                if (lbl >= 0 && lbl < K) {
                    buff_x[lbl] += img->x[idx];
                    buff_y[lbl] += img->y[idx];
                    buff_L[lbl] += img->L[idx];
                    buff_a[lbl] += img->A[idx];
                    buff_b[lbl] += img->B[idx];
                    buff_count[lbl]++;
                }
            }

#pragma omp for schedule(runtime)
            for (int k = 0; k < K; k++) {
                if (buff_count[k] > 0) {
                    super_pixels->centroid_x[k] = (int) (buff_x[k] / buff_count[k]);
                    super_pixels->centroid_y[k] = (int) (buff_y[k] / buff_count[k]);
                    super_pixels->val_L[k] = (float) (buff_L[k] / buff_count[k]);
                    super_pixels->val_a[k] = (float) (buff_a[k] / buff_count[k]);
                    super_pixels->val_b[k] = (float) (buff_b[k] / buff_count[k]);
                }
            }
        }
    }else {

#pragma omp parallel for schedule(runtime)
        for (int i = 0; i < this-> N; i++) {
            int lbl= img->labels[i];
            if (lbl >=0 && lbl < K) {
#pragma omp atomic
                buff_L[lbl] += img->L[i];
#pragma omp atomic
                buff_a[lbl] += img->A[i];
#pragma omp atomic
                buff_b[lbl] += img->B[i];
#pragma omp atomic
                buff_x[lbl] += img->x[i];
#pragma omp atomic
                buff_y[lbl] += img->y[i];
#pragma omp atomic
                buff_count[lbl]++;
            }

        }

#pragma omp parallel for schedule(runtime)
        for (int k = 0; k < K; k++) {
            if (buff_count[k] > 0) {
                super_pixels->centroid_x[k] = (int) (buff_x[k] / buff_count[k]);
                super_pixels->centroid_y[k] = (int) (buff_y[k] / buff_count[k]);
                super_pixels->val_L[k] = (float) (buff_L[k] / buff_count[k]);
                super_pixels->val_a[k] = (float) (buff_a[k] / buff_count[k]);
                super_pixels->val_b[k] = (float) (buff_b[k] / buff_count[k]);
            }
        }


    }
}

