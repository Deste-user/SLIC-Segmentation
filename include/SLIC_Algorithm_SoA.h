#ifndef SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_SOA_H
#define SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_SOA_H
#include "SLIC_common.h"
#include <opencv2/opencv.hpp>

struct Pixels_SoA {
    double *L = nullptr;
    double *A = nullptr;
    double *B = nullptr;
    double *distances = nullptr;
    int *x = nullptr;
    int *y = nullptr;
    int *labels = nullptr;

    ~Pixels_SoA() {
        delete[] L;
        delete[] A;
        delete[] B;
        delete[] distances;
        delete[] x;
        delete[] y;
        delete[] labels;
    }
};

struct SuperPixel_SoA {
    int* label = nullptr;
    int *centroid_x = nullptr;
    int *centroid_y = nullptr;
    double* val_L = nullptr;
    double* val_a = nullptr;
    double* val_b = nullptr;

    ~SuperPixel_SoA() {
        delete[] label;
        delete[] centroid_x;
        delete[] centroid_y;
        delete[] val_L;
        delete[] val_a;
        delete[] val_b;
    }
};

class SLIC_Algorithm_SoA: public SLIC_Algorithm {
protected:
    Pixels_SoA* img = nullptr;
    SuperPixel_SoA* super_pixels = nullptr;
public:
    SLIC_Algorithm_SoA() = default;
    int EnforceConnectivity() override;
    cv::Mat display_boundaries() override;
    float calculate_gradient(int x, int y) override;
    void clear() override;
    ~SLIC_Algorithm_SoA() override {
        delete img;
        delete super_pixels;
        free(this->buff_x);
        free(this->buff_y);
        free(this->buff_L);
        free(this->buff_a);
        free(this->buff_b);
        free(this->buff_count);
    }

};


#endif //SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_SOA_H