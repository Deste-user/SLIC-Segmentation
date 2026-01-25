#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <random>
#include "SLIC_common.h"
#include "SLIC_Algorithm_AoS_Sequential.h"

void SLIC_Algorithm_AoS_Sequential::Initialization() {
    int idx = 0;
    int i=0;
    // Regular grid
    for (int y = S/2 ; y < this->image_lab.rows; y += S) {
        for (int x = S/2 ; x < this->image_lab.cols; x += S) {
            if (i >= K) break;
            idx= x + this->image_lab.cols*y;
            this->spxls[i].centroid_x = x;
            this->spxls[i].centroid_y = y;
            this->spxls[i].val_L = pxls[idx].L;
            this->spxls[i].val_a = pxls[idx].A;
            this->spxls[i].val_b = pxls[idx].B;
            i++;
        }
    }

    // To adjust centroids to the lowest gradient position in a 3x3 neighborhood
    for (int k=0; k<K; k++) {
        float min_gradient= FLT_MAX;
        int best_x= this->spxls[k].centroid_x;
        int best_y= this->spxls[k].centroid_y;
        for (int dy=-1; dy<= 1; dy++) {
            for (int dx=-1; dx<=1;dx++) {
                int ny= this->spxls[k].centroid_y + dy;
                int nx= this->spxls[k].centroid_x + dx;
                if (nx > 0 && nx < this->image_lab.cols - 1 && ny > 0 && ny < this->image_lab.rows - 1)
                {
                    float g= this->calculate_gradient(nx, ny);
                    if (g < min_gradient) {
                        min_gradient= g;
                        best_x= nx;
                        best_y= ny;
                    }
                }
            }
        }
        this->spxls[k].centroid_x= best_x;
        this->spxls[k].centroid_y= best_y;
        idx= best_y*this->image_lab.cols + best_x;
        this->spxls[k].val_L= this->pxls[idx].L;
        this->spxls[k].val_a= this->pxls[idx].A;
        this->spxls[k].val_b= this->pxls[idx].B;
    }
}

void SLIC_Algorithm_AoS_Sequential:: iteration() {
    //int pixels_updated = 0;
    for (int i = 0; i < this->N; i++) {
        pxls[i].distance = DBL_MAX;
    }

    for (int k=0; k<K;k++) {
        int x_min = std::max(spxls[k].centroid_x - S, 0);
        int x_max = std::min(spxls[k].centroid_x + S, this->image_lab.cols);
        int y_min = std::max(spxls[k].centroid_y - S, 0);
        int y_max = std::min(spxls[k].centroid_y + S, this->image_lab.rows);

        for (int y=y_min; y<y_max;y++) {
            for (int x=x_min;x<x_max;x++) {
                int idx= x + this->image_lab.cols*y;
                double d = distance_SLIC(
                    spxls[k].val_L, spxls[k].val_a, spxls[k].val_b,
                    spxls[k].centroid_x, spxls[k].centroid_y,
                    pxls[idx].L, pxls[idx].A, pxls[idx].B,
                    pxls[idx].x, pxls[idx].y,
                    S, m);
                if (d < pxls[idx].distance) {
                    pxls[idx].distance= d;
                    pxls[idx].label= spxls[k].label;
                    //pixels_updated++;
                }
            }
        }
        /* TO VISUALIZE THE ITERATIONS
         *cv::Mat img = this->display_boundaries();
        cv::imshow("img before iteration", img);
        cv::waitKey(0);
        */
    }
    //std::cout << "[DEBUG] Pixels updated: " << pixels_updated << std::endl;
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



