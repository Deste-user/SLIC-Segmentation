#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <filesystem>

/* TODO: non usare gli smart pointers */

struct SuperPixel {
    int label = -1;
    int centroid_x = 0; // Colonna
    int centroid_y = 0; // Riga
    int val_L = 0;
    int val_a = 0;
    int val_b = 0;
};

// Funzione gradiente corretta (y, x)
int calculate_gradient(const cv::Mat& img, int x, int y) {
    int gradient = 0;
    for (int c = 0; c < 3; c++) {
        int diff_x = img.at<cv::Vec3b>(y, x + 1)[c] - img.at<cv::Vec3b>(y, x - 1)[c];
        int diff_y = img.at<cv::Vec3b>(y + 1, x)[c] - img.at<cv::Vec3b>(y - 1, x)[c];
        gradient += diff_x * diff_x + diff_y * diff_y;
    }
    return gradient;
}

double distance_SLIC(const SuperPixel& sp, int pxl_L, int pxl_a, int pxl_b, int pxl_x, int pxl_y, double m, double S) {
    double d_color = std::sqrt(std::pow(sp.val_L - pxl_L, 2) +
                               std::pow(sp.val_a - pxl_a, 2) +
                               std::pow(sp.val_b - pxl_b, 2));

    double d_spatial = std::sqrt(std::pow(sp.centroid_x - pxl_x, 2) +
                                 std::pow(sp.centroid_y - pxl_y, 2));

    return std::sqrt(std::pow(d_color, 2) + std::pow(d_spatial / S * m, 2));
}

class SLIC_Segmentation {
public:
    SLIC_Segmentation(const cv::Mat& image_lab, int K, double m = 10.0) {
        this->K = K;
        this->m = m;
        this->image_lab = image_lab;
        this->rows = image_lab.rows;
        this->cols = image_lab.cols;
        this->S = static_cast<int>(std::sqrt(static_cast<double>(rows * cols) / K));

        this->labels = cv::Mat(rows, cols, CV_32S, cv::Scalar(-1));
        this->distances = cv::Mat(rows, cols, CV_64F, cv::Scalar(DBL_MAX));
    }

    void initialization() {
        int lbl = 0;
        // Griglia regolare
        for (int y = S / 2; y < rows; y += S) {
            for (int x = S / 2; x < cols; x += S) {
                SuperPixel sp;
                sp.label = lbl++;
                sp.centroid_x = x;
                sp.centroid_y = y;
                cv::Vec3b color = image_lab.at<cv::Vec3b>(y, x);
                sp.val_L = color[0];
                sp.val_a = color[1];
                sp.val_b = color[2];
                superpixels.push_back(sp);
            }
        }

        // Spostamento su gradiente minimo (3x3)
        for (auto& sp : superpixels) {
            int min_gradient = INT_MAX;
            int best_x = sp.centroid_x;
            int best_y = sp.centroid_y;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ny = sp.centroid_y + dy;
                    int nx = sp.centroid_x + dx;
                    if (nx > 0 && nx < cols - 1 && ny > 0 && ny < rows - 1) {
                        int g = calculate_gradient(image_lab, nx, ny);
                        if (g < min_gradient) {
                            min_gradient = g;
                            best_x = nx;
                            best_y = ny;
                        }
                    }
                }
            }
            sp.centroid_x = best_x;
            sp.centroid_y = best_y;
            cv::Vec3b color = image_lab.at<cv::Vec3b>(best_y, best_x);
            sp.val_L = color[0]; sp.val_a = color[1]; sp.val_b = color[2];
        }
        this->realK = superpixels.size();
    }

    void iteration() {
        distances.setTo(cv::Scalar(DBL_MAX));
        for (const auto& sp : superpixels) {
            int x_min = std::max(0, sp.centroid_x - S);
            int x_max = std::min(cols, sp.centroid_x + S);
            int y_min = std::max(0, sp.centroid_y - S);
            int y_max = std::min(rows, sp.centroid_y + S);

            for (int y = y_min; y < y_max; y++) {
                for (int x = x_min; x < x_max; x++) {
                    cv::Vec3b pxl = image_lab.at<cv::Vec3b>(y, x);
                    double d = distance_SLIC(sp, pxl[0], pxl[1], pxl[2], x, y, m, S);

                    if (d < distances.at<double>(y, x)) {
                        distances.at<double>(y, x) = d;
                        labels.at<int>(y, x) = sp.label;
                    }
                }
            }
        }
    }

    void update_centroids() {
        std::vector<double> sum_x(realK, 0), sum_y(realK, 0), sum_L(realK, 0), sum_a(realK, 0), sum_b(realK, 0);
        std::vector<int> count(realK, 0);

        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                int lbl = labels.at<int>(y, x);
                if (lbl >= 0 && lbl < realK) {
                    cv::Vec3b pxl = image_lab.at<cv::Vec3b>(y, x);
                    sum_L[lbl] += pxl[0]; sum_a[lbl] += pxl[1]; sum_b[lbl] += pxl[2];
                    sum_x[lbl] += x; sum_y[lbl] += y;
                    count[lbl]++;
                }
            }
        }

        for (int i = 0; i < realK; i++) {
            if (count[i] > 0) {
                superpixels[i].centroid_x = static_cast<int>(sum_x[i] / count[i]);
                superpixels[i].centroid_y = static_cast<int>(sum_y[i] / count[i]);
                superpixels[i].val_L = static_cast<int>(sum_L[i] / count[i]);
                superpixels[i].val_a = static_cast<int>(sum_a[i] / count[i]);
                superpixels[i].val_b = static_cast<int>(sum_b[i] / count[i]);
            }
        }
    }

    void run(int num_iterations = 10) {
        initialization();
        for (int i = 0; i < num_iterations; i++) {
            iteration();
            update_centroids();
        }
    }

    void display_boundaries(cv::Mat& output) {
        for (int y = 1; y < rows - 1; y++) {
            for (int x = 1; x < cols - 1; x++) {
                if (labels.at<int>(y, x) != labels.at<int>(y, x + 1) ||
                    labels.at<int>(y, x) != labels.at<int>(y + 1, x)) {
                    output.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
                }
            }
        }
    }

private:
    int K, realK, S, rows, cols;
    double m;
    cv::Mat image_lab, labels, distances;
    std::vector<SuperPixel> superpixels;
};

int main() {
    cv::Mat image = cv::imread("/Users/marcodestefano/CLionProjects/SLIC Segmentation Algorithm/archive/images/test/100007.jpg");
    if (image.empty()) return -1;

    cv::Mat image_lab;
    cv::cvtColor(image, image_lab, cv::COLOR_BGR2Lab);

    SLIC_Segmentation slic(image_lab, 400, 30.0); // K=400, m=20
    slic.run(10);

    cv::Mat result = image.clone();
    slic.display_boundaries(result);

    cv::imshow("SLIC Result", result);
    cv::waitKey(0);
    return 0;
}

