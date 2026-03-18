#include <opencv2/opencv.hpp>
#include "SLIC_Algorithm_AoS_Parallel.h"
#include <omp.h>


void SLIC_Algorithm_AoS_Parallel::Initialization() {
    const int rows= this->image_lab.rows;
    const int cols= this->image_lab.cols;
    // Number of grid cells in the horizontal direction.
    const int grid_w= cols/S;
#pragma omp parallel
    {
        // To initialize the centroids of superpixels in a grid pattern.
    #pragma omp for schedule(runtime)
        for (int k=0; k<K;k++) {
            // Calculate index grid
            int grid_x= k % grid_w;
            int grid_y= k / grid_w;

            // Calculate Superpixel position in the grid - center pixel.
            int x = grid_x * S + S/2;
            int y = grid_y * S + S/2;
            // Checks if the center of the S Grid is in the image.
            if (x < cols && y < rows) {
                int idx = x + y * cols;
                spxls[k].val_L= pxls[idx].L;
                spxls[k].val_a= pxls[idx].A;
                spxls[k].val_b= pxls[idx].B;
                spxls[k].centroid_x= pxls[idx].x;
                spxls[k].centroid_y= pxls[idx].y;
            }
        }

    // To adjust centroids to the lowest gradient position in a 3x3 neighborhood
    #pragma omp for schedule(runtime)
        for (int k = 0; k < this->K; k++) {
            float min_gradient = FLT_MAX;
            int best_x = spxls[k].centroid_x;
            int best_y = spxls[k].centroid_y;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ny = spxls[k].centroid_y + dy;
                    int nx = spxls[k].centroid_x + dx;
                    if (nx > 0 && nx < cols - 1 && ny > 0 && ny < rows - 1) {
                        float g = calculate_gradient(nx, ny);
                        if (g < min_gradient) {
                            min_gradient = g;
                            best_x = nx;
                            best_y = ny;
                        }
                    }
                }
            }
            spxls[k].centroid_x = best_x;
            spxls[k].centroid_y = best_y;
            int idx = best_x + cols * best_y;
            spxls[k].val_L = pxls[idx].L;
            spxls[k].val_a = pxls[idx].A;
            spxls[k].val_b = pxls[idx].B;
        }
    }
}

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
                                if (abs(spxls[k].centroid_x - x) < 2 * S &&
                                    abs(spxls[k].centroid_y - y) < 2 * S) {
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
#pragma omp for collapse(2) schedule(runtime)
            for (int ty = 0; ty < rows; ty += TILE_SIZE) {
                for (int tx = 0; tx < cols; tx += TILE_SIZE) {
                    int y_end = std::min(ty + TILE_SIZE, rows);
                    int x_end = std::min(tx + TILE_SIZE, cols);

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
                                        if (abs(spxls[k].centroid_x - x) < 2 * S &&
                                            abs(spxls[k].centroid_y - y) < 2 * S) {
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

    // 2. FIRST TOUCH: Inizializziamo a zero IN PARALLELO
    // Così ogni core alloca fisicamente la memoria che userà nella sua cache locale (NUMA friendly)
#pragma omp parallel for schedule(static)
    for(int i=0; i<K; i++) {
        buff_x[i] = 0.0;
        buff_y[i] = 0.0;
        buff_L[i] = 0.0;
        buff_a[i] = 0.0;
        buff_b[i] = 0.0;
        buff_count[i] = 0;
    }
    if (this->reduction_parallel) {
#pragma omp parallel
        {
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

#pragma omp for schedule(runtime)
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
    } else {
#pragma omp parallel for schedule(static)
        for (int idx = 0; idx < this->N; idx++) {
            int lbl = pxls[idx].label;
            if (lbl >= 0 && lbl < K) {
                // Aggiornamento atomico: blocca solo la specifica cella di memoria,
                // non ferma tutto il thread. Molto veloce se le collisioni sono poche.
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

#pragma omp parallel for schedule(static)
        for (int k = 0; k < K; k++) {
            if (buff_count[k] > 0) {
                spxls[k].centroid_x = (int)(buff_x[k] / buff_count[k]);
                spxls[k].centroid_y = (int)(buff_y[k] / buff_count[k]);
                spxls[k].val_L = (float)(buff_L[k] / buff_count[k]);
                spxls[k].val_a = (float)(buff_a[k] / buff_count[k]);
                spxls[k].val_b = (float)(buff_b[k] / buff_count[k]);
            }
        }
    }
}
