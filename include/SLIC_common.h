#ifndef SLIC_SEGMENTATION_ALGORITHM_SLIC_COMMON_H
#define SLIC_SEGMENTATION_ALGORITHM_SLIC_COMMON_H
#include <opencv2/opencv.hpp>
#include <string>

#define TILE_SIZE 32
#define PATH_images "../archive/images/val/"
#define PATH_images_test "../archive/images/test/"
#define PATH_example "../archive/images/test/2018.jpg"
#include <opencv2/opencv.hpp>

enum DataLayout {
    AoS,
    SoA
};

class SLIC_Algorithm{
public:
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
        int new_K = EnforceConnectivity();
        if (new_K < 1) new_K = 1;
        if (new_K > K_max) new_K = K_max;
        this->K = new_K;
        this->update_centroids();
    };
    void set_tiling(const bool val) {
        use_tiling = val;
    }

    void set_image_lab(const cv::Mat &image_lab) {
        this->image_lab = image_lab;
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
    void set_K(int new_K) {
        this->K = new_K;
        double N = (double)this->image_lab.rows * this->image_lab.cols;
        this->S = (int)sqrt(N / new_K);
        int cols_steps = image_lab.cols / S;
        int rows_steps = image_lab.rows / S;
        this->K = rows_steps * cols_steps;
    }

    bool use_tiling{false};
protected:
    cv::Mat image_lab;
    int K{0};
    int K_max{0};
    int N{0};
    int S{0};
    int m{0};
    int num_iterations{0};
    double *buff_x = nullptr;
    double *buff_y = nullptr;
    double *buff_L = nullptr;
    double *buff_a = nullptr;
    double *buff_b = nullptr;
    int *buff_count = nullptr;

};

// Utilità condivise
std::string get_random_image_path(const std::string& folder_path);
double distance_SLIC(float cL, float cA, float cB, int cx, int cy,
                     float pL, float pA, float pB, int px, int py, int S, int m);

#endif //SLIC_SEGMENTATION_ALGORITHM_SLIC_COMMON_H