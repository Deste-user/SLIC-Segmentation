#ifndef SLIC_SEGMENTATION_ALGORITHM_SLIC_COMMON_H
#define SLIC_SEGMENTATION_ALGORITHM_SLIC_COMMON_H
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <string>

#define TILE_SIZE 64
#define PATH_images "../archive/images/test/"
#define PATH_example "/Users/marcodestefano/CLionProjects/SLIC Segmentation Algorithm/archive/images/test/2018.jpg"

enum DataLayout {
    AoS,
    SoA
};

class SLIC_Algorithm{
public:
    bool use_tiling= false;
    virtual ~SLIC_Algorithm() = default;
    virtual std::string get_name() const = 0;
    virtual DataLayout get_data_layout() const = 0;
    virtual bool is_parallel() const = 0;
    virtual void Initialization()=0;
    virtual cv::Mat display_boundaries()=0;
    virtual void  iteration()=0;
    virtual void update_centroids()=0;
    virtual int EnforceConnectivity()=0;
    virtual float calculate_gradient(int x, int y)=0;
    virtual void clear()=0;
    void run() {
        this->Initialization();
        for (int i = 0; i < num_iterations; i++) {
            this->iteration();
            this->update_centroids();
        }
        this->K = EnforceConnectivity();
        this->update_centroids();
    };
    void set_tiling(const bool val) {
        use_tiling = val;
    }

    bool use_tiling1() const {
        return use_tiling;
    }

    cv::Mat image_lab1() const {
        return image_lab;
    }

    int k() const {
        return K;
    }

    int n() const {
        return N;
    }

    int s() const {
        return S;
    }

    int m1() const {
        return m;
    }

    int num_iterations_get() const {
        return num_iterations;
    }
protected:
    cv::Mat image_lab;
    int K;
    int K_max;
    int N;
    int S;
    int m;
    int num_iterations;
};

// Utilità condivise
std::string get_random_image_path(const std::string& folder_path);
double distance_SLIC(float cL, float cA, float cB, int cx, int cy,
                     float pL, float pA, float pB, int px, int py, int S, int m);

#endif //SLIC_SEGMENTATION_ALGORITHM_SLIC_COMMON_H