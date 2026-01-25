#ifndef SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_SOA_SEQUENTIAL_H
#define SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_SOA_SEQUENTIAL_H
#include "SLIC_Algorithm_SoA.h"
#include <cfloat>

class SLIC_Algorithm_SoA_Sequential: public SLIC_Algorithm_SoA {
public:
    SLIC_Algorithm_SoA_Sequential(cv::Mat image_lab, int K, int m, int iterations) {
        this->image_lab= image_lab;
        this->N = image_lab.cols * image_lab.rows;
        this->K = K;
        this->m = m;
        this->num_iterations = iterations;
        this->S = (int) std::sqrt((double) (image_lab.rows * image_lab.cols) / K);
        int cols_steps = image_lab.cols / S;
        int rows_steps = image_lab.rows / S;
        this->K = rows_steps * cols_steps;
        this->K_max = 2*this->K;
        this->img= new Pixels_SoA();
        this->super_pixels= new SuperPixel_SoA();

        img->L = (double*) malloc(N * sizeof(double));
        img->A = (double*) malloc(N * sizeof(double));
        img->B = (double*) malloc(N * sizeof(double));
        img->x = (int*) malloc(N * sizeof(int));
        img->y = (int*) malloc(N * sizeof(int));
        img->distances = (double*) malloc(N * sizeof(double));
        img->labels = (int*) malloc(N * sizeof(int));

        int cols = image_lab.cols;
        int rows = image_lab.rows;

        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {

                int i = y * cols + x; // Indice lineare univoco

                cv::Vec3b pixel = image_lab.at<cv::Vec3b>(y, x);

                img->L[i] = (double)pixel[0];
                img->A[i] = (double)pixel[1];
                img->B[i] = (double)pixel[2];

                img->x[i] = x;
                img->y[i] = y;
                img->distances[i] = DBL_MAX;
                img->labels[i] = -1;
            }
        }
        this->super_pixels->label = (int*) malloc(this->K_max * sizeof(int));
        this->super_pixels->centroid_x = (int*) malloc(this->K_max * sizeof(int));
        this->super_pixels->centroid_y = (int*) malloc(this->K_max * sizeof(int));
        this->super_pixels->val_L = (double*) malloc(this->K_max * sizeof(double));
        this->super_pixels->val_a = (double*) malloc(this->K_max * sizeof(double));
        this->super_pixels->val_b = (double*) malloc(this->K_max * sizeof(double));

        for (int l = 0; l < this->K; l++) {
            super_pixels->label[l] = l;
        }

        this->buff_L = (double*) malloc(this->K_max * sizeof(double));
        this->buff_a = (double*) malloc(this->K_max * sizeof(double));
        this->buff_b = (double*) malloc(this->K_max * sizeof(double));
        this->buff_x = (double*) malloc(this->K_max * sizeof(double));
        this->buff_y = (double*) malloc(this->K_max * sizeof(double));
        this->buff_count = (int*) malloc(this->K_max * sizeof(int));
    }

    void Initialization() override;
    void iteration() override;
    void update_centroids() override;

    std::string get_name() const override {return "SOA Sequential SLIC";}
    DataLayout get_data_layout() const override {return DataLayout::SoA;}
    bool is_parallel() const override {return false;}

};
#endif //SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_SOA_SEQUENTIAL_H