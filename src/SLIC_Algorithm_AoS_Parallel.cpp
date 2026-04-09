#include <opencv2/opencv.hpp>
#include "SLIC_Algorithm_AoS_Parallel.h"
#include <omp.h>


void SLIC_Algorithm_AoS_Parallel::iteration() {
    const int rows = this->image_lab.rows;
    const int cols = this->image_lab.cols;
    const int grid_w = cols / S;
    const int grid_h = rows / S;
#pragma omp parallel
    {
        if (!this->use_tiling) {
#pragma omp  for schedule(runtime)
            for (int y = 0; y < rows; y++) {
                int grid_y = y / S;
                for (int x = 0; x < cols; x++) {
                    int idx = x + cols * y;
                    int grid_x = x / S;

                    float val_L = pxls[idx].L;
                    float val_A = pxls[idx].A;
                    float val_B = pxls[idx].B;
                    int pos_x = pxls[idx].x;
                    int pos_y = pxls[idx].y;

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
                            if (k >= 0 && k < K) {
                                if (abs(spxls[k].centroid_x - x) < S &&
                                    abs(spxls[k].centroid_y - y) < S) {
                                    double d = distance_SLIC(spxls[k].val_L, spxls[k].val_a, spxls[k].val_b,
                                                             spxls[k].centroid_x, spxls[k].centroid_y,
                                                             val_L, val_A, val_B, pos_x, pos_y, S, m);

                                    if (d < min_distance) {
                                        min_distance = d;
                                        best_k = k;
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                    if (changed) {
                        pxls[idx].distance = (float) min_distance;
                        pxls[idx].label = best_k;
                    }
                }
            }
        } else {
            int current_tile_size = 2 * this->S;
#pragma omp for collapse(2) schedule(runtime)
            for (int ty = 0; ty < rows; ty += current_tile_size) {
                for (int tx = 0; tx < cols; tx += current_tile_size) {
                    int y_end = std::min(ty + current_tile_size, rows);
                    int x_end = std::min(tx + current_tile_size, cols);

                    for (int y = ty; y < y_end; y++) {
                        int grid_y = y / S;
                        for (int x = tx; x < x_end; x++) {
                            int idx = x + cols * y;

                            int grid_x = x / S;
                            float val_L = pxls[idx].L;
                            float val_A = pxls[idx].A;
                            float val_B = pxls[idx].B;
                            int pos_x = pxls[idx].x;
                            int pos_y = pxls[idx].y;

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

                                    if (k >= 0 && k < K) {
                                        if (abs(spxls[k].centroid_x - x) < S &&
                                            abs(spxls[k].centroid_y - y) < S) {
                                            double d = distance_SLIC(spxls[k].val_L, spxls[k].val_a, spxls[k].val_b,
                                                                     spxls[k].centroid_x,
                                                                     spxls[k].centroid_y, val_L, val_A, val_B, pos_x,
                                                                     pos_y, S, m);

                                            if (d < min_distance) {
                                                min_distance = d;
                                                best_k = k;
                                                changed = true;
                                            }
                                        }
                                    }
                                }
                            }

                            if (changed) {
                                pxls[idx].distance = (float) min_distance;
                                pxls[idx].label = best_k;
                            }
                        }
                    }
                }
            }
        }
    }
}


void SLIC_Algorithm_AoS_Parallel::update_centroids() {
// NUMA friendly
#pragma omp parallel for schedule(static)
    for(int i=0; i<K; i++) {
        buff_x[i] = 0.0;
        buff_y[i] = 0.0;
        buff_L[i] = 0.0;
        buff_a[i] = 0.0;
        buff_b[i] = 0.0;
        buff_count[i] = 0;
    }

#pragma omp parallel
    {

        if (this->reduction_parallel) {
#pragma omp for schedule(runtime) \
reduction(+: buff_x[:K], buff_y[:K], buff_L[:K], \
buff_a[:K], buff_b[:K], buff_count[:K])
            for (int idx = 0; idx < N; idx++) {
                int lbl = pxls[idx].label;
                if (lbl >= 0 && lbl < K) {
                    buff_L[lbl] += pxls[idx].L;
                    buff_a[lbl] += pxls[idx].A;
                    buff_b[lbl] += pxls[idx].B;
                    buff_x[lbl] += pxls[idx].x;
                    buff_y[lbl] += pxls[idx].y;
                    buff_count[lbl]++;
                }
            }
        } else {
#pragma omp for schedule(runtime)
            for (int idx = 0; idx < this->N; idx++) {
                int lbl = pxls[idx].label;
                if (lbl >= 0 && lbl < K) {
#pragma omp atomic
                    buff_L[lbl] += pxls[idx].L;
#pragma omp atomic
                    buff_a[lbl] += pxls[idx].A;
#pragma omp atomic
                    buff_b[lbl] += pxls[idx].B;
#pragma omp atomic
                    buff_x[lbl] += pxls[idx].x;
#pragma omp atomic
                    buff_y[lbl] += pxls[idx].y;
#pragma omp atomic
                    buff_count[lbl]++;
                }
            }
        }
#pragma omp for simd schedule(runtime)
        for (int k = 0; k < K; k++) {
            int count = buff_count[k];
            if (count > 0) {
                double inv = 1.0 / count;
                spxls[k].centroid_x = (int)   (buff_x[k] * inv);
                spxls[k].centroid_y = (int)   (buff_y[k] * inv);
                spxls[k].val_L      = (float) (buff_L[k] * inv);
                spxls[k].val_a      = (float) (buff_a[k] * inv);
                spxls[k].val_b      = (float) (buff_b[k] * inv);
            }
        }
    }
}

