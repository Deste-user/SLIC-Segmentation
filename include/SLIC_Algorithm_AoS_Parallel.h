#include "SLIC_Algorithm_AoS.h"

#ifndef SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_AOS_PARALLEL_H
#define SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_AOS_PARALLEL_H

class SLIC_Algorithm_AoS_Parallel: public SLIC_Algorithm_AoS {
public:
    SLIC_Algorithm_AoS_Parallel(cv::Mat image_lab, int K, int m, int iterations) {
        this->N = image_lab.cols * image_lab.rows;
        this->K = K;
        this->m = m;
        this->image_lab = image_lab;
        this->num_iterations = iterations;
        this->pxls = (Pixel_AoS *) malloc(N * sizeof(Pixel_AoS));
        this->S = (int) std::sqrt((double) (image_lab.rows * image_lab.cols) / K);
        this->K_max = this->K;
        int cols_steps = image_lab.cols / S;
        int rows_steps = image_lab.rows / S;
        this->K = rows_steps * cols_steps;
        this->spxls = (SuperPixel_AoS *) malloc(this->K * sizeof(SuperPixel_AoS));

        unsigned char* raw_data = image_lab.data;
        // Linearization of the Image.
#pragma omp parallel for collapse(2)
        for (int y = 0; y < image_lab.rows; y++) {
            for (int x = 0; x < image_lab.cols; x++) {
                int idx = y * image_lab.cols + x;
                int img_idx = idx * 3;  // 3 canali (L, A, B)

                this->pxls[idx].L = (float)raw_data[img_idx];
                this->pxls[idx].A = (float)raw_data[img_idx + 1];
                this->pxls[idx].B = (float)raw_data[img_idx + 2];
                this->pxls[idx].x = x;
                this->pxls[idx].y = y;
                this->pxls[idx].distance = FLT_MAX;
                this->pxls[idx].label = -1;
            }
        }

        //To inizialize the labels of all super pixels
#pragma omp parallel for schedule(static)
        for (int i = 0; i < this->K; i++) {
            this->spxls[i].label = i;
        }

    }
    void Initialization() override;
    void iteration() override;
    void update_centroids() override;

    std::string get_name() const override {return "AOS Parallel SLIC";}
    DataLayout get_data_layout() const override {return DataLayout::AoS;}
    bool is_parallel() const override {return true;}
};

#endif //SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_AOS_PARALLEL_H
