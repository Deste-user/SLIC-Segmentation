// In this file we implement the parallel version of SLIC using Structure of Arrays (SoA) approach.
#include "SLIC_common.h"
#include <opencv2/opencv.hpp>

#include <omp.h>
#include "SLIC_Algorithm_SoA_Parallel.h"


void SLIC_Algorithm_SoA_Parallel::Initialization() {
    int grid_w = this->image_lab.cols/S;
    int grid_h = this->image_lab.rows / S;
#pragma omp parallel default(none) shared(grid_w, grid_h)
    {
        #pragma omp for schedule(runtime)
        for (int i =0; i < K; i++) {
            int grid_x = i % grid_w;
            int grid_y = i / grid_w;
            if (grid_y >= grid_h) continue;

            int x = grid_x * S + S / 2;
            int y = grid_y * S + S / 2;

            x = std::min(x, this->image_lab.cols - 1);
            y = std::min(y, this->image_lab.rows - 1);

            super_pixels->centroid_x[i] = x;
            super_pixels->centroid_y[i] = y;

            int idx = y * this->image_lab.cols + x;

            if (idx >= 0 && idx < N) {
                super_pixels->val_L[i] = img->L[idx];
                super_pixels->val_a[i] = img->A[idx];
                super_pixels->val_b[i] = img->B[idx];
            }
        }

        // Spostamento su gradiente minimo (3x3)
        #pragma omp for schedule(runtime)
        for (int k = 0; k < K; k++) {
            float min_gradient = FLT_MAX;
            int best_x = super_pixels->centroid_x[k];
            int best_y = super_pixels->centroid_y[k];

            // Small cycle to parallelize
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ny = super_pixels->centroid_y[k] + dy;
                    int nx = super_pixels->centroid_x[k] + dx;
                    if (nx > 0 && nx < this->image_lab.cols - 1 && ny > 0 && ny < this->image_lab.rows - 1) {
                        float g = calculate_gradient(nx, ny);
                        if (g < min_gradient) {
                            min_gradient = g;
                            best_x = nx;
                            best_y = ny;
                        }
                    }
                }
            }
            super_pixels->centroid_x[k] = best_x;
            super_pixels->centroid_y[k] = best_y;
            int idx = best_y * this->image_lab.cols + best_x;
            super_pixels->val_L[k] = img->L[idx];
            super_pixels->val_a[k] = img->A[idx];
            super_pixels->val_b[k] = img->B[idx];
        }
    }
}

// Use the pixel centric version - each pixel checks all superpixels in its 2Sx2S regio
void SLIC_Algorithm_SoA_Parallel::iteration() {
    // I define these variables here to avoid recalculating them in the loops
    const int rows = this->image_lab.rows;
    const int cols = this->image_lab.cols;
    const int grid_w = cols/S;
#pragma omp parallel
    {
        if(!this->use_tiling) {
            #pragma omp for schedule(runtime)
                for (int y = 0; y < rows; y++) {
                    int grid_y = y / S;
                    for (int x=0; x < cols; x++){
                        int idx = x + cols * y;
                        int grid_x = x / S;

                        // We can save here in local variables to reduce memory accesses
                        float val_L = img->L[idx];
                        float val_a = img->A[idx];
                        float val_b = img->B[idx];
                        int pos_x = img->x[idx];
                        int pos_y = img->y[idx];
                        double min_distance = DBL_MAX;
                        int best_k = img->labels[idx];
                        bool changed = false;


                        for (int ny = -1; ny <= 1; ny++) {
                            //I can save some computations out of the inner loop
                            int ky = grid_y + ny;
                            int k_row_offset = ky * grid_w;
                            for (int nx = -1; nx <= 1; nx++) {
                                int kx = grid_x + nx;
                                int k = k_row_offset + kx;
                                if (k >= 0 && k < K) {
                                    if (abs(super_pixels->centroid_x[k] - x) < 2 * S &&
                                        abs(super_pixels->centroid_y[k] - y) < 2 * S) {
                                        double d = distance_SLIC(super_pixels->val_L[k], super_pixels->val_a[k],
                                                                 super_pixels->val_b[k], super_pixels->centroid_x[k],
                                                                 super_pixels->centroid_y[k],
                                                                 val_L, val_a,val_b,pos_x,pos_y, S,
                                                                 m);

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
                            img->distances[idx] =(float) min_distance;
                            img->labels[idx] = best_k;
                        }
                    }
                }
        }else{
            #pragma omp for collapse(2) schedule(runtime)
            // When I acccess to a location, I could load a block of data into the cache.
            // Therefore, I access contiguous blocks of data to make the best use of the cache.
            // I suppose that the TILE_SIZE is 64.
            for (int by = 0; by < rows; by += TILE_SIZE) {
                for (int bx = 0; bx < cols; bx += TILE_SIZE) {
                    int y_end = std::min(by + TILE_SIZE, this->image_lab.rows);
                    int x_end = std::min(bx + TILE_SIZE, this->image_lab.cols);
                    // Sfrutto la cache.
                    for (int y = by; y < y_end; y++) {
                        int grid_y = y / S;
                        for (int x = bx; x < x_end; x++) {
                            int idx = x + cols * y;
                            int grid_x = x / S;

                            float val_L = img->L[idx];
                            float val_a = img->A[idx];
                            float val_b = img->B[idx];
                            float pos_x = img->x[idx];
                            float pos_y = img->y[idx];

                            double min_distance = img->distances[idx];
                            int best_k = img->labels[idx];
                            bool changed = false;

                            for (int ny = -1; ny <= 1; ny++) {
                                int ky = grid_y + ny;
                                int k_row_offset = ky * grid_w;
                                for (int nx = -1; nx <= 1; nx++) {
                                    int kx = grid_x + nx;

                                    int k = k_row_offset + kx;
                                    if (k >= 0 && k < K) {
                                        if (abs(super_pixels->centroid_x[k] - x) < 2 * S &&
                                            abs(super_pixels->centroid_y[k] - y) < 2 * S) {
                                            double d = distance_SLIC(super_pixels->val_L[k], super_pixels->val_a[k],
                                                                     super_pixels->val_b[k], super_pixels->centroid_x[k],
                                                                     super_pixels->centroid_y[k],
                                                                     val_L,val_a,val_b,pos_x,pos_y, S, m);

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
    int max_threads = omp_get_max_threads();
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
buff_a[:K_max], buff_b[:K_max], buff_count[:K_max])     // si usa nowait poichè non serve aspettare gli altri.
        for (int idx = 0; idx < N; idx++) {
            int lbl = img->labels[idx];
            if (lbl >= 0 && lbl < K_max) {
                buff_x[lbl] += img->x[idx];
                buff_y[lbl] += img->y[idx];
                buff_L[lbl] += img->L[idx];
                buff_a[lbl] += img->A[idx];
                buff_b[lbl] += img->B[idx];
                buff_count[lbl]++;
            }
        }
        // Implicit Barrier has sum all the arrays.

        // Now each thread can update a portion of the centroids.
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
        if (lbl >=0 && lbl < K_max) {
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
