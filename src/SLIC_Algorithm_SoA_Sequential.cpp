#include <opencv2/opencv.hpp>
#include "SLIC_common.h"
#include "SLIC_Algorithm_SoA_Sequential.h"
/*
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
*/
void SLIC_Algorithm_SoA_Sequential::iteration() {
    const int rows = this->image_lab.rows;
    const int cols = this->image_lab.cols;
    const int grid_w = cols / S;
    const int grid_h = rows / S;

    // Scorrimento Pixel-Centric: iteriamo su tutti i pixel dell'immagine
    for (int y = 0; y < rows; y++) {
        int grid_y = y / S;
        for (int x = 0; x < cols; x++) {
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