#ifndef SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_AOS_H
#define SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_AOS_H
#include "SLIC_common.h"

struct SuperPixel_AoS {
    int label;
    int centroid_x; // Colonna
    int centroid_y; // Riga
    float val_L;
    float val_a;
    float val_b;
};

struct Pixel_AoS {
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
        Pixel_AoS* pxls;
        SuperPixel_AoS* spxls;
    public:
        std::string get_name() const override { return "AoS Parallel SLIC"; }
        DataLayout get_data_layout() const override { return DataLayout::AoS; }
        bool is_parallel() const override { return true; }
        int EnforceConnectivity() override ;
        void clear() override;
        float calculate_gradient(int x, int y) override;
        cv::Mat display_boundaries() override;
        ~SLIC_Algorithm_AoS() override{
            free(this->pxls);
            free(this->spxls);
        };
};


#endif //SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_AOS_H