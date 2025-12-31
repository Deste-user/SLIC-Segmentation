#ifndef SLIC_SEGMENTATION_ALGORITHM_SLIC_COMMON_H
#define SLIC_SEGMENTATION_ALGORITHM_SLIC_COMMON_H
#include <opencv2/opencv.hpp>
#include <string>

#define TILE_SIZE 64
#define PATH_images "/Users/marcodestefano/CLionProjects/SLIC Segmentation Algorithm/archive/images/test/"
#define PATH_example "/Users/marcodestefano/CLionProjects/SLIC Segmentation Algorithm/archive/images/test/2018.jpg"

enum DataLayout {
    AoS,
    SoA
};

class SLIC_Algorithm {
public:
    bool use_tiling= false;
    virtual ~SLIC_Algorithm() = default;
    virtual std::string get_name() const = 0;
    virtual DataLayout get_data_layout() const = 0;
    virtual bool is_parallel() const = 0;
    virtual void Initialization() {};
    virtual void run() {};
    virtual cv::Mat display_boundaries() {};
    virtual void  iteration() {};
    virtual void update_centroids() {};
    virtual int EnforceConnectivity() {return 0;};
    virtual float calculate_gradient(int x, int y) {return 0;};
};

// Utilità condivise
std::string get_random_image_path(const std::string& folder_path);
double distance_SLIC(float cL, float cA, float cB, int cx, int cy,
                     float pL, float pA, float pB, int px, int py, int S, int m);

#endif //SLIC_SEGMENTATION_ALGORITHM_SLIC_COMMON_H