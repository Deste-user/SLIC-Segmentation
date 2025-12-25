#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <random>
#define PATH_images "/Users/marcodestefano/CLionProjects/SLIC Segmentation Algorithm/archive/images/test/"

namespace fs = std::filesystem;

struct SuperPixels {
    int* label;
    int *centroid_x; // Colonna
    int *centroid_y; // Riga
    float* val_L;
    float* val_a;
    float* val_b;
};


struct Image {
    float *L, *A, *B, *distances;
    int *x,*y, *labels;
};


int calculate_gradient(Image* img, int x, int y, int cols) {
    int gradient = 0;

    int idx = y * cols + x;
    int idx_right = y * cols + (x + 1);
    int idx_left = y * cols + (x - 1);
    int idx_down = (y + 1) * cols + x;
    int idx_up = (y - 1) * cols + x;

    // Differences L
    float diff_x_L = img->L[idx_right] - img->L[idx_left];
    float diff_y_L = img->L[idx_down] - img->L[idx_up];

    // Differences A
    float diff_x_A = img->A[idx_right] - img->A[idx_left];
    float diff_y_A = img->A[idx_down] - img->A[idx_up];

    // Differences B
    float diff_x_B = img->B[idx_right] - img->B[idx_left];
    float diff_y_B = img->B[idx_down] - img->B[idx_up];

    gradient = (int) (diff_x_L * diff_x_L + diff_y_L * diff_y_L +
               diff_x_A * diff_x_A + diff_y_A * diff_y_A +
               diff_x_B * diff_x_B + diff_y_B * diff_y_B);

    return gradient;
}

double distance_SLIC(float centroid_L,float centroid_A, float centroid_B, int centroid_x, int centroid_y, float pxl_L,
    float pxl_A,float pxl_B, int pxl_x, int pxl_y,int S, int m) {
    // Calculate the distance of LAB
    double d_color= std::sqrt(
        (centroid_L - pxl_L) * (centroid_L - pxl_L) +
        (centroid_A - pxl_A) * (centroid_A - pxl_A) +
        (centroid_B - pxl_B) * (centroid_B - pxl_B)
    );

    // Calcola distanza spaziale
    double d_spatial = std::sqrt(
        (centroid_x - pxl_x) * (centroid_x - pxl_x) +
        (centroid_y - pxl_y) * (centroid_y - pxl_y)
    );

    // Distance SLIC
    double d = std::sqrt(d_color * d_color + (d_spatial / S * m) * (d_spatial / S * m));
    return d;
}

int Inizialization_structures(Image* img, SuperPixels* super_pixels, cv::Mat image_lab,int K) {
    int N=image_lab.cols*image_lab.rows;


    img->L = (float*) malloc(N * sizeof(float));
    img->A = (float*) malloc(N * sizeof(float));
    img->B = (float*) malloc(N * sizeof(float));
    img->x = (int*) malloc(N * sizeof(int));
    img->y = (int*) malloc(N * sizeof(int));
    img->distances = (float*) malloc(N * sizeof(float));
    img->labels = (int*) malloc(N * sizeof(int));

    int idx = 0;

    //Linearization of the Image.
    for (int y=0; y< image_lab.rows; y++) {
        for (int x=0; x< image_lab.cols; x++) {
            cv::Vec3b pixel = image_lab.at<cv::Vec3b>(y, x);
            img->L[idx] = pixel[0];
            img->A[idx] = pixel[1];
            img->B[idx] = pixel[2];
            img->x[idx] = x;
            img->y[idx] = y;
            img->distances[idx]= MAXFLOAT;
            img->labels[idx]=-1;
            idx++;
        }
    }
    //Linearization of all super pixels
    super_pixels->centroid_x = (int*)malloc(K * sizeof(int));
    super_pixels->centroid_y = (int*)malloc(K * sizeof(int));
    super_pixels->val_L = (float*)malloc(K * sizeof(float));
    super_pixels->val_a = (float*)malloc(K * sizeof(float));
    super_pixels->val_b = (float*)malloc(K * sizeof(float));
    super_pixels->label = (int*)malloc(K * sizeof(int));

    for (int l=0; l < K; l++) {
        super_pixels->label[l] = l;
    }

    return ((int) std::sqrt(N/K));
}

void Initialization(Image* img,SuperPixels* sp ,int S, int rows, int cols, int K) {
    int idx = 0;
    int i=0;
    // Griglia regolare
    for (int y = S/2 ; y < rows; y += S) {
        for (int x = S/2 ; x < cols; x += S) {
            if (i >= K) break;
            idx= x + cols*y;
            sp->centroid_x[i] = x;
            sp->centroid_y[i] = y;
            sp->val_L[i] = img->L[idx];
            sp->val_a[i] = img->A[idx];
            sp->val_b[i] = img->B[idx];
            i++;
        }
        if (i >= K) break;
    }

    // Spostamento su gradiente minimo (3x3)
    for (int k=0 ; k < K; k++) {
        int min_gradient = INT_MAX;
        int best_x = sp->centroid_x[k];
        int best_y = sp->centroid_y[k];

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int ny = sp->centroid_y[k] + S*dy;
                int nx = sp->centroid_x[k] + S*dx;
                if (nx > 0 && nx < cols - 1 && ny > 0 && ny < rows - 1) {
                    int g = calculate_gradient(img, nx, ny,cols);
                    if (g < min_gradient) {
                        min_gradient = g;
                        best_x = nx;
                        best_y = ny;
                    }
                }
            }
        }
        sp->centroid_x[k] = best_x;
        sp->centroid_y[k] = best_y;
        idx = best_y*cols+best_x;
        sp->val_L[k] = img->L[idx];
        sp->val_a[k] = img->A[idx];
        sp->val_b[k] = img->B[idx];
    }
}

void iteration(Image* img, SuperPixels* sp, int S, int rows, int cols, int K, int m) {
    // Reset all distances
    for (int i = 0; i < rows * cols; i++) {
        img->distances[i] = MAXFLOAT;
        img->labels[i] = -1;
    }

    // For all superpixel
    for (int k = 0; k < K; k++) {
        // Track the movement in the 2Sx2S region
        int x_min = std::max(0, sp->centroid_x[k] - S);
        int x_max = std::min(cols, sp->centroid_x[k] + S);
        int y_min = std::max(0, sp->centroid_y[k] - S);
        int y_max = std::min(rows, sp->centroid_y[k] + S);

        // For all pixel in the region 2S x 2S
        for (int y = y_min; y < y_max; y++) {
            for (int x = x_min; x < x_max; x++) {
                int idx = y * cols + x;

                double d = distance_SLIC(
                    sp->val_L[k], sp->val_a[k], sp->val_b[k],
                    sp->centroid_x[k], sp->centroid_y[k],
                    img->L[idx], img->A[idx], img->B[idx],
                    img->x[idx], img->y[idx],
                    S, m);
                // Update if the distance is smaller
                if (d < img->distances[idx]) {
                    img->distances[idx] = d;
                    img->labels[idx] = k;
                }
            }
        }
    }
}


void update_centroids(Image* img, SuperPixels* sp, int rows, int cols, int K) {
    // Array per accumulatori
    double* sum_x = (double*)calloc(K, sizeof(double));
    double* sum_y = (double*)calloc(K, sizeof(double));
    double* sum_L = (double*)calloc(K, sizeof(double));
    double* sum_a = (double*)calloc(K, sizeof(double));
    double* sum_b = (double*)calloc(K, sizeof(double));
    int* count = (int*)calloc(K, sizeof(int));

    // Accumula valori per ogni superpixel
    for (int idx = 0; idx < rows * cols; idx++) {
        int lbl = img->labels[idx];
        if (lbl >= 0 && lbl < K) {
            sum_L[lbl] += img->L[idx];
            sum_a[lbl] += img->A[idx];
            sum_b[lbl] += img->B[idx];
            sum_x[lbl] += img->x[idx];
            sum_y[lbl] += img->y[idx];
            count[lbl]++;
        }
    }

    // Calcola nuovi centroidi
    for (int k = 0; k < K; k++) {
        if (count[k] > 0) {
            sp->centroid_x[k] = (int)(sum_x[k] / count[k]);
            sp->centroid_y[k] = (int)(sum_y[k] / count[k]);
            sp->val_L[k] = (float)(sum_L[k] / count[k]);
            sp->val_a[k] = (float)(sum_a[k] / count[k]);
            sp->val_b[k] = (float)(sum_b[k] / count[k]);
        }
    }

    // Libera memoria
    free(sum_x);
    free(sum_y);
    free(sum_L);
    free(sum_a);
    free(sum_b);
    free(count);
}


void run(Image* img, SuperPixels* sp,int S,int row,int cols,int K,int num_iterations = 10) {
    Initialization(img, sp, S, row, cols, K);
    for (int i = 0; i < num_iterations; i++) {
        iteration(img,sp, S, row, cols, K, 10);
        update_centroids(img, sp, row, cols, K);
    }
}


cv::Mat display_boundaries(Image* output,SuperPixels* sp, int rows, int cols) {
    int idx = 0;
    cv::Mat lab_mat(rows, cols, CV_8UC3);
    for (int y = 0; y < rows - 1; y++) {
        for (int x = 0; x < cols - 1; x++) {
            idx = y * cols + x;
            lab_mat.at<cv::Vec3b>(y, x)[0] = (uchar)sp->val_L[output->labels[idx]];
            lab_mat.at<cv::Vec3b>(y, x)[1] = (uchar)sp->val_a[output->labels[idx]];
            lab_mat.at<cv::Vec3b>(y, x)[2] = (uchar)sp->val_b[output->labels[idx]];
        }
    }
    cv::Mat output_mat;
    cv::cvtColor(lab_mat, output_mat, cv::COLOR_Lab2BGR);

    for (int y = 0; y < rows ; y++) {
        for (int x = 0; x < cols ; x++) {
            idx = y * cols + x;
            int idx_right = y * cols + (x + 1);
            int idx_down = (y + 1) * cols + x;

            // Se il pixel ha un vicino con label diversa, disegna bordo nero
            if (output->labels[idx] != output->labels[idx_right] ||
                output->labels[idx] != output->labels[idx_down]) {
                output_mat.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);  // Nero
                }
        }
    }
    return output_mat;
}
cv::Mat display_boundaries1(Image* img, SuperPixels* sp, int rows, int cols) {
    // Crea immagine con colori medi dei superpixel
    cv::Mat lab_mat(rows, cols, CV_8UC3);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            int idx = y * cols + x;
            int label = img->labels[idx];

            if (label >= 0) {
                lab_mat.at<cv::Vec3b>(y, x)[0] = (uchar)sp->val_L[label];
                lab_mat.at<cv::Vec3b>(y, x)[1] = (uchar)sp->val_a[label];
                lab_mat.at<cv::Vec3b>(y, x)[2] = (uchar)sp->val_b[label];
            }
        }
    }

    // Converti da LAB a BGR
    cv::Mat output;
    cv::cvtColor(lab_mat, output, cv::COLOR_Lab2BGR);

    // Disegna bordi neri
    for (int y = 0; y < rows - 1; y++) {
        for (int x = 0; x < cols - 1; x++) {
            int idx = y * cols + x;
            int idx_right = idx + 1;
            int idx_down = idx + cols;

            // Controlla se il pixel è sul bordo di un superpixel
            if (img->labels[idx] != img->labels[idx_right] ||
                img->labels[idx] != img->labels[idx_down]) {
                output.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
        }
    }

    // Gestisci ultima riga
    for (int x = 0; x < cols - 1; x++) {
        int idx = (rows - 1) * cols + x;
        if (img->labels[idx] != img->labels[idx + 1]) {
            output.at<cv::Vec3b>(rows - 1, x) = cv::Vec3b(0, 0, 0);
        }
    }

    // Gestisci ultima colonna
    for (int y = 0; y < rows - 1; y++) {
        int idx = y * cols + (cols - 1);
        if (img->labels[idx] != img->labels[idx + cols]) {
            output.at<cv::Vec3b>(y, cols - 1) = cv::Vec3b(0, 0, 0);
        }
    }

    return output;
}



// Funzione per ottenere un percorso immagine randomico
std::string get_random_image_path(const std::string& folder_path) {
    std::vector<std::string> valid_images;

    // Controlla se la cartella esiste
    if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
        std::cerr << "Errore: La cartella non esiste o il percorso è errato: " << folder_path << std::endl;
        return "";
    }

    // Itera nella cartella
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            // Controlla l'estensione per evitare file nascosti tipo .DS_Store
            std::string ext = entry.path().extension().string();
            // Converti in lowercase per sicurezza
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

    // Generatore randomico moderno (meglio di rand())
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, valid_images.size() - 1);

    int random_index = distrib(gen);
    return valid_images[random_index];
}

int main() {
    std::string img_path = get_random_image_path(PATH_images);

    cv::Mat image = cv::imread(img_path);
    if (image.empty()) return -1;
    cv::Mat image_lab;
    cv::cvtColor(image, image_lab, cv::COLOR_BGR2Lab);
    Image img{};
    SuperPixels superpixels{};
    int K=100;
    int S = Inizialization_structures(&img, &superpixels, image_lab, K);

    Initialization(&img, &superpixels,S,image_lab.rows, image_lab.cols,K);
    run(&img, &superpixels, S, image_lab.rows, image_lab.cols, K, 10);

    cv::Mat output= display_boundaries1(&img, &superpixels, image_lab.rows, image_lab.cols);

    free(img.L);
    free(img.A);
    free(img.B);
    free(img.x);
    free(img.y);

    free(img.distances);
    free(img.labels);
    free(superpixels.centroid_x);
    free(superpixels.centroid_y);
    free(superpixels.val_L);
    free(superpixels.val_a);
    free(superpixels.val_b);
    free(superpixels.label);


    cv::imshow("SLIC Result", output);
    cv::waitKey(0);

    return 0;
}

