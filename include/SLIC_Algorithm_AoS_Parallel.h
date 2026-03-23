#ifndef SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_AOS_PARALLEL_H
#define SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_AOS_PARALLEL_H

#include "SLIC_Algorithm_AoS.h"


class SLIC_Algorithm_AoS_Parallel: public SLIC_Algorithm_AoS {
public:
    // Constructor that initializes the SLIC Algorithm
    SLIC_Algorithm_AoS_Parallel(cv::Mat image_lab, int K, int m, int iterations,bool reduction_parallel= true) {
        this->N = image_lab.cols * image_lab.rows;
        this->K = K;
        this->m = m;
        this->reduction_parallel = reduction_parallel;
        this->image_lab = image_lab;
        this->num_iterations = iterations;
        //Allocate memory for the pixels and superpixels
        this->pxls = (Pixel_AoS *) malloc(N * sizeof(Pixel_AoS));
        // Calculate the grid step S.
        this->S = (int) std::sqrt((double) (image_lab.rows * image_lab.cols) / K);
        // Recalculate K based on the actual grid size.
        int cols_steps = image_lab.cols / S;
        int rows_steps = image_lab.rows / S;
        this->K = rows_steps * cols_steps;
        // Keep extra capacity for connectivity enforcement splits.
        this->K_max = 2 * this->K;
        this->spxls = (SuperPixel_AoS *) malloc(this->K_max * sizeof(SuperPixel_AoS));


        // Linearization of the Image.
        for (int y = 0; y < image_lab.rows; y++) {
            for (int x = 0; x < image_lab.cols; x++) {
                int idx = y * image_lab.cols + x;
                cv::Vec3b pixel = image_lab.at<cv::Vec3b>(y, x);

                this->pxls[idx].L = (float)pixel[0];
                this->pxls[idx].A = (float)pixel[1];
                this->pxls[idx].B = (float)pixel[2];
                this->pxls[idx].x = x;
                this->pxls[idx].y = y;
                this->pxls[idx].distance = DBL_MAX;
                this->pxls[idx].label = -1;
            }
        }

        //To inizialize the labels of all super pixels - with a id from 0 to K-1
        for (int i = 0; i < this->K; i++) {
            this->spxls[i].label = i;
        }

        //Allocate the buffers for the accumulation.
        this->buff_x = (double *) malloc(this->K_max * sizeof(double));
        this->buff_y = (double *) malloc(this->K_max * sizeof(double));
        this->buff_L = (double *) malloc(this->K_max * sizeof(double));
        this->buff_a = (double *) malloc(this->K_max * sizeof(double));
        this->buff_b = (double *) malloc(this->K_max * sizeof(double));
        this->buff_count = (int *) malloc(this->K_max * sizeof(int));
    }
    bool reduction_parallel;
    void iteration() override;
    void update_centroids() override;

    std::string get_name() const override {return "AOS Parallel SLIC";}
    DataLayout get_data_layout() const override {return DataLayout::AoS;}
    bool is_parallel() const override {return true;}
};

#endif //SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_AOS_PARALLEL_H
