#ifndef SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_SOA_H
#define SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_SOA_H
#include "SLIC_common.h"

struct SuperPixel_SoA{
    int* label;
    int *centroid_x;
    int *centroid_y;
    float* val_L;
    float* val_a;
    float* val_b;
};

struct Pixels_SoA{
    float *L, *A, *B, *distances;
    int *x,*y, *labels;
};


class SLIC_Algorithm_SoA: public SLIC_Algorithm {
protected:
    Pixels_SoA* img;
    SuperPixel_SoA* super_pixels;
public:
    int EnforceConnectivity() override;
    cv::Mat display_boundaries() override;
    float calculate_gradient(int x, int y) override;
    void clear() override;
    ~SLIC_Algorithm_SoA() {
        if (img) {
            free(img->L);
            free(img->A);
            free(img->B);
            free(img->x);
            free(img->y);
            free(img->distances);
            free(img->labels);
            delete img;
        }

        if (super_pixels) {
            free(super_pixels->label);
            free(super_pixels->centroid_x);
            free(super_pixels->centroid_y);
            free(super_pixels->val_L);
            free(super_pixels->val_a);
            free(super_pixels->val_b);
            delete super_pixels;
        }
    }
};


#endif //SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_SOA_H