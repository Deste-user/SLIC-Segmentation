#include <opencv2/opencv.hpp>
#include "SLIC_common.h"
#include "SLIC_Algorithm_SoA_Sequential.h"


void SLIC_Algorithm_SoA_Sequential:: Initialization() {
    int idx = 0;
    int i=0;
    // Griglia regolare
    for (int y = S/2 ; y < this->image_lab.rows; y += S) {
        for (int x = S/2 ; x < this->image_lab.cols; x += S) {
            if (i >= K) break;
            idx= x + this->image_lab.cols*y;
            super_pixels->centroid_x[i] = x;
            super_pixels->centroid_y[i] = y;
            super_pixels->val_L[i] = img->L[idx];
            super_pixels->val_a[i] = img->A[idx];
            super_pixels->val_b[i] = img->B[idx];
            i++;
        }
        if (i >= K) break;
    }

    // Spostamento su gradiente minimo (3x3)
    for (int k=0 ; k < K; k++) {
        float min_gradient = FLT_MAX;
        int best_x = super_pixels->centroid_x[k];
        int best_y = super_pixels->centroid_y[k];

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ny = super_pixels->centroid_y[k] + dy;
                int nx = super_pixels->centroid_x[k] + dx;
                if (nx > 0 && nx < this->image_lab.cols - 1 && ny > 0 && ny < this->image_lab.rows - 1) {
                    float g = this->calculate_gradient(nx, ny);
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

        idx = best_y*this->image_lab.cols+best_x;
        super_pixels->val_L[k] = img->L[idx];
        super_pixels->val_a[k] = img->A[idx];
        super_pixels->val_b[k] = img->B[idx];
    }
}

void SLIC_Algorithm_SoA_Sequential:: iteration() {
    // Reset all distances
    for (int i = 0; i < N; i++) {
        img->distances[i] = MAXFLOAT;
        img->labels[i] = -1;
    }

    // For all superpixel
    for (int k = 0; k < K; k++) {
        // Track the movement in the 2Sx2S region
        int x_min = std::max(0, super_pixels->centroid_x[k] - S);
        int x_max = std::min(this->image_lab.cols, super_pixels->centroid_x[k] + S);
        int y_min = std::max(0, super_pixels->centroid_y[k] - S);
        int y_max = std::min(this->image_lab.rows, super_pixels->centroid_y[k] + S);

        // For all pixel in the region 2S x 2S
        for (int y = y_min; y < y_max; y++) {
            for (int x = x_min; x < x_max; x++) {
                int idx = y * this->image_lab.cols + x;

                double d = distance_SLIC(
                    super_pixels->val_L[k], super_pixels->val_a[k], super_pixels->val_b[k],
                    super_pixels->centroid_x[k], super_pixels->centroid_y[k],
                    img->L[idx], img->A[idx], img->B[idx],
                    img->x[idx], img->y[idx],
                    S, m);
                // Update if the distance is smaller
                if (d < img->distances[idx]) {
                    img->distances[idx] = d;
                    img->labels[idx] = k;
                }
            }
        }
    }
}


void SLIC_Algorithm_SoA_Sequential:: update_centroids() {

    for (int k = 0; k < K_max; k++) {
        buff_L[k] = 0.0;
        buff_a[k] = 0.0;
        buff_b[k] = 0.0;
        buff_x[k] = 0.0;
        buff_y[k] = 0.0;
        buff_count[k] = 0;
    }

    // Accumula valori per ogni superpixel
    for (int idx = 0; idx < N; idx++) {
        int lbl = img->labels[idx];
        if (lbl >= 0 && lbl < K) {
            this->buff_L[lbl] += img->L[idx];
            this->buff_a[lbl] += img->A[idx];
            this->buff_b[lbl] += img->B[idx];
            this->buff_x[lbl] += img->x[idx];
            this->buff_y[lbl] += img->y[idx];
            this->buff_count[lbl]++;
        }
    }

    // Calcola nuovi centroidi
    for (int k = 0; k < K; k++) {
        if (buff_count[k] > 0) {
            super_pixels->centroid_x[k] = (int)(buff_x[k] / buff_count[k]);
            super_pixels->centroid_y[k] = (int)(buff_y[k] / buff_count[k]);
            super_pixels->val_L[k] = (float)(buff_L[k] / buff_count[k]);
            super_pixels->val_a[k] = (float)(buff_a[k] / buff_count[k]);
            super_pixels->val_b[k] = (float)(buff_b[k] / buff_count[k]);
        }
    }

}