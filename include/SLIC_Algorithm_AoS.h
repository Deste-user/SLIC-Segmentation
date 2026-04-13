#ifndef SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_AOS_H
#define SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_AOS_H
#include "SLIC_common.h"
#include <opencv2/opencv.hpp>
#include <new>

struct SuperPixel_AoS {
    int label;
    int centroid_x;
    int centroid_y;
    float val_L;
    float val_a;
    float val_b;
};

struct alignas(std::hardware_destructive_interference_size) Pixel_AoS {
    int label;
    int x;
    int y;
    float distance;
    float L;
    float A;
    float B;
};

class SLIC_Algorithm_AoS: public SLIC_Algorithm {
    protected:
        Pixel_AoS* pxls = nullptr;
        SuperPixel_AoS* spxls = nullptr;
    public:
        std::string get_name() const override { return "AoS Parallel SLIC"; }
        DataLayout get_data_layout() const override { return DataLayout::AoS; }
        bool is_parallel() const override { return true; }
        void Initialization() override;
        int EnforceConnectivity() override ;
        void clear() override;
        float calculate_gradient(int x, int y) override;
        cv::Mat display_boundaries() override;
        ~SLIC_Algorithm_AoS() override{
            free(this->pxls);
            free(this->spxls);
            free(this->buff_x);
            free(this->buff_y);
            free(this->buff_L);
            free(this->buff_a);
            free(this->buff_b);
            free(this->buff_count);
        };
};


#endif //SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_AOS_H