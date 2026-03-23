#ifndef SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_SOA_PARALLEL_H
#define SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_SOA_PARALLEL_H
#include "SLIC_Algorithm_SoA.h"
#include <cfloat>
#include <omp.h>
class SLIC_Algorithm_SoA_Parallel: public SLIC_Algorithm_SoA {
public:
    SLIC_Algorithm_SoA_Parallel(cv::Mat image_lab, int K, int m, int iterations, bool reduction_parallel=true) {
        this->image_lab= image_lab;
        this->N = image_lab.cols * image_lab.rows;
        this->K = K;
        this->m = m;
        this->num_iterations = iterations;
        this->reduction_parallel = reduction_parallel;

        this->S = (int) std::sqrt((double) (image_lab.rows * image_lab.cols) / K);
        int cols_steps = image_lab.cols / S;
        int rows_steps = image_lab.rows / S;
        this->K = rows_steps * cols_steps;
        this->K_max = 2 * this->K;
        this->img= new Pixels_SoA();
        this->super_pixels= new SuperPixel_SoA();

        img->L = new double[N];
        img->A = new double[N];
        img->B = new double[N];
        img->x = new int[N];
        img->y = new int[N];
        img->labels = new int[N];
        img->distances = new double[N];


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


        // Initialization super pixels, double of K for the Enhance Connectivity
        super_pixels->centroid_x = new int[K_max];
        super_pixels->centroid_y = new int[K_max];
        super_pixels->val_L = new double[K_max];
        super_pixels->val_a = new double[K_max];
        super_pixels->val_b = new double[K_max];
        super_pixels->label = new int[K_max];

        // This cycle is too small to parallelize (overhead > guadagno).
        for (int l = 0; l < this->K_max; l++) {
            super_pixels->label[l] = (l < K) ? l : -1;
            super_pixels->centroid_x[l] = 0;
            super_pixels->centroid_y[l] = 0;
            super_pixels->val_L[l] = 0.0f;
            super_pixels->val_a[l] = 0.0f;
            super_pixels->val_b[l] = 0.0f;
        }

        this->buff_x = (double *) malloc(this->K_max * sizeof(double));
        this->buff_y = (double *) malloc(this->K_max * sizeof(double));
        this->buff_L = (double *) malloc(this->K_max * sizeof(double));
        this->buff_a = (double *) malloc(this->K_max * sizeof(double));
        this->buff_b = (double *) malloc(this->K_max * sizeof(double));
        this->buff_count = (int *) malloc(this->K_max * sizeof(int));

    }

    bool reduction_parallel;
    ~SLIC_Algorithm_SoA_Parallel() override = default;
    void iteration() override;
    void update_centroids() override;


    std::string get_name() const override {return "SOA Parallel SLIC";}
    DataLayout get_data_layout() const override {return DataLayout::SoA;}
    bool is_parallel() const override {return true;}
};


#endif //SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_SOA_PARALLEL_H