#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <random>
#include "SLIC_common.h"
#include "SLIC_Algorithm_AoS_Sequential.h"


void SLIC_Algorithm_AoS_Sequential:: iteration() {
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

            float val_L = pxls[idx].L;
            float val_a = pxls[idx].A;
            float val_b = pxls[idx].B;
            int pos_x = pxls[idx].x;
            int pos_y = pxls[idx].y;

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
                    if (k < 0 || k >= K) continue;

                    if (abs(spxls[k].centroid_x - x) < S &&
                        abs(spxls[k].centroid_y - y) < S) {

                        double d = distance_SLIC(spxls[k].val_L, spxls[k].val_a,
                                                 spxls[k].val_b, spxls[k].centroid_x,
                                                 spxls[k].centroid_y,
                                                 val_L, val_a, val_b, pos_x, pos_y, S, m);
                        if (d < min_distance) {
                            min_distance = d;
                            best_k = k;
                        }
                    }
                }
            }

            if (best_k >= 0) {
                pxls[idx].distance = (float) min_distance;
                pxls[idx].label = best_k;
            }
        }
    }
}

void SLIC_Algorithm_AoS_Sequential:: update_centroids() {

    for (int k = 0; k < K; k++) {
        this->buff_x[k] = 0.0;
        this->buff_y[k] = 0.0;
        this->buff_L[k] = 0.0;
        this->buff_a[k] = 0.0;
        this->buff_b[k] = 0.0;
        this->buff_count[k] = 0;
    }


    // Accumulate values for each superpixel
    for (int idx = 0; idx < this->N; idx++) {
        int lbl = pxls[idx].label;
        if (lbl >= 0 && lbl < K) {
            this->buff_L[lbl] += pxls[idx].L;
            this->buff_a[lbl] += pxls[idx].A;
            this->buff_b[lbl] += pxls[idx].B;
            this->buff_x[lbl] += pxls[idx].x;
            this->buff_y[lbl] += pxls[idx].y;
            this->buff_count[lbl]++;
        }
    }

    // Calculate new centroids
    for (int k = 0; k < K; k++) {
        if (this->buff_count[k] > 0) {
            spxls[k].centroid_x = (int)(this->buff_x[k] / this->buff_count[k]);
            spxls[k].centroid_y = (int)(this->buff_y[k] / this->buff_count[k]);
            spxls[k].val_L = (float)(this->buff_L[k] / this->buff_count[k]);
            spxls[k].val_a = (float)(this->buff_a[k] / this->buff_count[k]);
            spxls[k].val_b = (float)(this->buff_b[k] / this->buff_count[k]);
        }
    }
}



