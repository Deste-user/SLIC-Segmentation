#ifndef SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_AOS_SEQUENTIAL_H
#define SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_AOS_SEQUENTIAL_H
#include "SLIC_Algorithm_AoS.h"

class SLIC_Algorithm_AoS_Sequential: public SLIC_Algorithm_AoS {
    public:
        SLIC_Algorithm_AoS_Sequential(cv::Mat image_lab, int K, int m, int iterations) {
            int idx = 0;

            this->N = image_lab.cols * image_lab.rows;
            this->K = K;
            this->m = m;
            this->image_lab = image_lab;
            this->num_iterations = iterations;
            this->pxls = (Pixel_AoS *) malloc(N * sizeof(Pixel_AoS));
            this->S = (int) std::sqrt((double) (image_lab.rows * image_lab.cols) / K);
            int cols_steps = image_lab.cols / S;
            int rows_steps = image_lab.rows / S;
            this->K = rows_steps * cols_steps;
            this->K_max = this->K;
            this->spxls = (SuperPixel_AoS *) malloc(this->K * sizeof(SuperPixel_AoS));

            // Linearization of the Image.
            for (int y = 0; y < image_lab.rows; y++) {
                for (int x = 0; x < image_lab.cols; x++) {
                    idx= y * image_lab.cols + x;
                    cv::Vec3b pixel = image_lab.at<cv::Vec3b>(y, x);
                    pxls[idx].distance = MAXFLOAT;
                    pxls[idx].label = -1;
                    pxls[idx].L = pixel[0];
                    pxls[idx].A = pixel[1];
                    pxls[idx].B = pixel[2];
                    pxls[idx].x = x;
                    pxls[idx].y = y;
                }
            }

            //To inizialize the labels of all super pixels
            for (int i=0; i < this->K; i++) {
                spxls[i].label = i;
            }
        }
        void Initialization() override;
        void iteration() override;
        void update_centroids() override;

        std::string get_name() const override {return "AOS Sequential SLIC";}
        DataLayout get_data_layout() const override {return DataLayout::AoS;}
        bool is_parallel() const override {return false;}

};


#endif //SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_AOS_SEQUENTIAL_H