#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <random>
#define PATH_images "/Users/marcodestefano/CLionProjects/SLIC Segmentation Algorithm/archive/images/test/"

struct SuperPixel {
    int label;
    int centroid_x; // Colonna
    int centroid_y; // Riga
    float val_L;
    float val_a;
    float val_b;
};
struct Pixel {
    int label;
    int x;
    int y;
    float distance;
    float L;
    float A;
    float B;
};

void Initialization_Structures(Pixel* pxls, cv::Mat image_lab, SuperPixel* spxs, int K) {
    int idx = 0;

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
    for (int i=0; i < K; i++) {
        spxs[i].label = i;
    }
}

float calculate_gradient(Pixel* pxls, int x, int y, int cols) {
    float gradient = 0.0f;

    int idx_right = y * cols + (x + 1);
    int idx_left = y * cols + (x - 1);
    int idx_down = (y + 1) * cols + x;
    int idx_up = (y - 1) * cols + x;

    // Differences L
    float diff_x_L = pxls[idx_right].L - pxls[idx_left].L;
    float diff_y_L = pxls[idx_down].L - pxls[idx_up].L;

    // Differences A
    float diff_x_A = pxls[idx_right].A - pxls[idx_left].A;
    float diff_y_A = pxls[idx_down].A - pxls[idx_up].A;

    // Differences B
    float diff_x_B = pxls[idx_right].B - pxls[idx_left].B;
    float diff_y_B = pxls[idx_down].B - pxls[idx_up].B;

    gradient = diff_x_L * diff_x_L + diff_y_L * diff_y_L +
               diff_x_A * diff_x_A + diff_y_A * diff_y_A +
               diff_x_B * diff_x_B + diff_y_B * diff_y_B;

    return gradient;
}

void Initialization(Pixel* pxls, SuperPixel* spxs ,int S, int rows, int cols, int K) {
    int idx = 0;
    int i=0;
    // Regular grid
    for (int y = S/2 ; y < rows; y += S) {
        for (int x = S/2 ; x < cols; x += S) {
            if (i >= K) break;
            idx= x + cols*y;
            spxs[i].centroid_x = x;
            spxs[i].centroid_y = y;
            spxs[i].val_L = pxls[idx].L;
            spxs[i].val_a = pxls[idx].A;
            spxs[i].val_b = pxls[idx].B;
            i++;
        }
    }

    // To adjust centroids to the lowest gradient position in a 3x3 neighborhood
    for (int k=0; k<K; k++) {
        float min_gradient= FLT_MAX;
        int best_x= spxs[k].centroid_x;
        int best_y= spxs[k].centroid_y;
        for (int dy=-1; dy< 1; dy++) {
            for (int dx=-1; dx<1;dx++) {
                int ny= spxs[k].centroid_y + S*dy;
                int nx= spxs[k].centroid_x + S*dx;
                if (nx > 0 || nx < cols -1 || ny > 0 || ny < rows -1) {
                    float g= calculate_gradient(pxls, nx, ny, cols);
                    if (g < min_gradient) {
                        min_gradient= g;
                        best_x= nx;
                        best_y= ny;
                    }
                }
            }
        }
        spxs[k].centroid_x= best_x;
        spxs[k].centroid_y= best_y;
    }
}

//Da vedere se mettere in un file.
double distance_SLIC(float centroid_L,float centroid_A, float centroid_B, int centroid_x, int centroid_y, float pxl_L,
    float pxl_A,float pxl_B, int pxl_x, int pxl_y,int S, int m) {
    // Calculate the distance of LAB
    double d_color= std::sqrt(
        (centroid_L - pxl_L) * (centroid_L - pxl_L) +
        (centroid_A - pxl_A) * (centroid_A - pxl_A) +
        (centroid_B - pxl_B) * (centroid_B - pxl_B)
    );

    // Calculate spatial distance
    double d_spatial = std::sqrt(
        (centroid_x - pxl_x) * (centroid_x - pxl_x) +
        (centroid_y - pxl_y) * (centroid_y - pxl_y)
    );

    // Distance SLIC
    double d = std::sqrt(d_color * d_color + (d_spatial / S * m) * (d_spatial / S * m));
    return d;
}

void iteration(Pixel* pxls, SuperPixel* spxls, int S, int rows, int cols, int K, int m) {

    for (int k=0; k<K;k++) {
        int x_min = std::max(spxls[k].centroid_x - S, 0);
        int x_max = std::min(spxls[k].centroid_x + S, cols);
        int y_min = std::max(spxls[k].centroid_y - S, 0);
        int y_max = std::min(spxls[k].centroid_y + S, rows);

        for (int y=y_min; y<y_max;y++) {
            for (int x=x_min;x<x_max;x++) {
                int idx= x + cols*y;
                double d = distance_SLIC(
                    spxls[k].val_L, spxls[k].val_a, spxls[k].val_b,
                    spxls[k].centroid_x, spxls[k].centroid_y,
                    pxls[idx].L, pxls[idx].A, pxls[idx].B,
                    pxls[idx].x, pxls[idx].y,
                    S, m);
                if (d < pxls[idx].distance) {
                    pxls[idx].distance= d;
                    pxls[idx].label= spxls[k].label;
                }
            }
        }
    }
}

void update_centroids(Pixel* pxls, SuperPixel* spxls, int rows, int columns, int K) {
    double* sum_x = (double*)calloc(K, sizeof(double));
    double* sum_y = (double*)calloc(K, sizeof(double));
    double* sum_L = (double*)calloc(K, sizeof(double));
    double* sum_a = (double*)calloc(K, sizeof(double));
    double* sum_b = (double*)calloc(K, sizeof(double));
    int* count = (int*)calloc(K, sizeof(int));

    // Accumulate values for each superpixel
    for (int idx = 0; idx < rows * columns; idx++) {
        int lbl = pxls[idx].label;
        if (lbl >= 0 && lbl < K) {
            sum_L[lbl] += pxls[idx].L;
            sum_a[lbl] += pxls[idx].A;
            sum_b[lbl] += pxls[idx].B;
            sum_x[lbl] += pxls[idx].x;
            sum_y[lbl] += pxls[idx].y;
            count[lbl]++;
        }
    }

    // Calculate new centroids
    for (int k = 0; k < K; k++) {
        if (count[k] > 0) {
            spxls[k].centroid_x = (int)(sum_x[k] / count[k]);
            spxls[k].centroid_y = (int)(sum_y[k] / count[k]);
            spxls[k].val_L = (float)(sum_L[k] / count[k]);
            spxls[k].val_a = (float)(sum_a[k] / count[k]);
            spxls[k].val_b = (float)(sum_b[k] / count[k]);
        }
    }

    // Free memory
    free(sum_x);
    free(sum_y);
    free(sum_L);
    free(sum_a);
    free(sum_b);
    free(count);
}



void run(Pixel* pxls, SuperPixel* spxls,int rows, int cols,int K, int m, int iterations) {
    int S = std::sqrt((rows * cols) / K);
    Initialization(pxls, spxls, S, rows, cols, K);
    for (int i = 0; i < iterations; i++) {
        iteration(pxls, spxls, S, rows, cols, K, m);
        update_centroids(pxls, spxls, rows, cols, K);
    }
}

// Vedere se mettere in un file a parte.
std::string get_random_image_path(const std::string& folder_path) {
    std::vector<std::string> valid_images;
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                valid_images.push_back(entry.path().string());
            }
        }
    }

    if (valid_images.empty()) {
        std::cerr << "Nessuna immagine trovata nella cartella!" << std::endl;
        return "";
    }

    // Modern random generator (better than rand())
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, valid_images.size() - 1);

    int random_index = distrib(gen);
    return valid_images[random_index];
}

cv::Mat display_boundaries(Pixel* pxls, SuperPixel* spxs, int rows, int cols) {
    cv::Mat lab_mat(rows, cols, CV_8UC3);
    for (int y = 0; y < rows ; y++) {
        for (int x = 0; x < cols ; x++) {
            int idx = y * cols + x;
            lab_mat.at<cv::Vec3b>(y, x)[0] = (uchar)spxs[pxls[idx].label].val_L;
            lab_mat.at<cv::Vec3b>(y, x)[1] = (uchar)spxs[pxls[idx].label].val_a;
            lab_mat.at<cv::Vec3b>(y, x)[2] = (uchar)spxs[pxls[idx].label].val_b;
        }
    }
    cv::Mat output_mat;
    cv::cvtColor(lab_mat, output_mat, cv::COLOR_Lab2BGR);

    // Draw boundaries
    for (int y = 0; y < rows - 1; y++) {
        for (int x = 0; x < cols - 1; x++) {
            int idx = y * cols + x;
            int idx_right = y * cols + (x + 1);
            int idx_down = (y + 1) * cols + x;

            if (pxls[idx].label != pxls[idx_right].label || pxls[idx].label != pxls[idx_down].label) {
                output_mat.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255); // Red boundary
            }
        }
    }

    return output_mat;
}

int main() {
    std::string img_path = get_random_image_path(PATH_images);

    cv::Mat image = cv::imread(img_path);
    if (image.empty()) return -1;
    cv::Mat image_lab;
    cv::cvtColor(image, image_lab, cv::COLOR_BGR2Lab);
    int K = 100;
    int N = image_lab.cols * image_lab.rows;
    Pixel* pxls= (Pixel*) malloc(image_lab.cols * image_lab.rows * sizeof(Pixel));
    SuperPixel* spxs= (SuperPixel*) malloc(K * sizeof(SuperPixel));
    Initialization_Structures(pxls, image_lab, spxs, K);
    run(pxls, spxs, image_lab.rows, image_lab.cols, K, 10, 10);
    cv::Mat output(image_lab.rows, image_lab.cols, CV_8UC3);
    output= display_boundaries(pxls, spxs, image_lab.rows, image_lab.cols);
    cv::imshow("SLIC Result AoS", output);
    cv::waitKey(0);

    free(pxls);
    free(spxs);

    return 0;
}

