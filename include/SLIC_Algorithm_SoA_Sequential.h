#ifndef SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_SOA_SEQUENTIAL_H
#define SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_SOA_SEQUENTIAL_H
#include "SLIC_Algorithm_SoA.h"

class SLIC_Algorithm_SoA_Sequential: public SLIC_Algorithm_SoA {
public:
    SLIC_Algorithm_SoA_Sequential(cv::Mat image_lab, int K, int m, int iterations) {
        this->image_lab= image_lab;
        this->N = image_lab.cols * image_lab.rows;
        this->K = K;
        this->m = m;
        this->num_iterations = iterations;
        this->S = (int) std::sqrt((double) (image_lab.rows * image_lab.cols) / K);
        int cols_steps = image_lab.cols / S;
        int rows_steps = image_lab.rows / S;
        this->K = rows_steps * cols_steps;
        this->K_max = this->K;
        this->img= new Pixels_SoA();
        this->super_pixels= new SuperPixel_SoA();

        img->L = (float*) malloc(N * sizeof(float));
        img->A = (float*) malloc(N * sizeof(float));
        img->B = (float*) malloc(N * sizeof(float));
        img->x = (int*) malloc(N * sizeof(int));
        img->y = (int*) malloc(N * sizeof(int));
        img->distances = (float*) malloc(N * sizeof(float));
        img->labels = (int*) malloc(N * sizeof(int));

        // Otteniamo il puntatore raw ai dati dell'immagine per velocità massima
        // Assumiamo che l'immagine sia continua in memoria (quasi sempre vero con imread)
        unsigned char* raw_data = image_lab.data;
        int cols = image_lab.cols;
        int rows = image_lab.rows;

        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {

                int i = y * cols + x; // Indice lineare univoco

                // Accesso diretto alla memoria OpenCV (molto più veloce di .at)
                // L'immagine LAB ha 3 canali. Struttura: [L, A, B, L, A, B...]
                int img_idx = i * 3;

                img->L[i] = (float)raw_data[img_idx];     // Canale 0
                img->A[i] = (float)raw_data[img_idx + 1]; // Canale 1
                img->B[i] = (float)raw_data[img_idx + 2]; // Canale 2

                img->x[i] = x;
                img->y[i] = y;
                img->distances[i] = MAXFLOAT;
                img->labels[i] = -1;
            }
        }
        this->super_pixels->label = (int*) malloc(this->K * sizeof(int));
        this->super_pixels->centroid_x = (int*) malloc(this->K * sizeof(int));
        this->super_pixels->centroid_y = (int*) malloc(this->K * sizeof(int));
        this->super_pixels->val_L = (float*) malloc(this->K * sizeof(float));
        this->super_pixels->val_a = (float*) malloc(this->K * sizeof(float));
        this->super_pixels->val_b = (float*) malloc(this->K * sizeof(float));

        for (int l = 0; l < this->K; l++) {
            super_pixels->label[l] = l;
        }
    }
    void Initialization() override;
    void iteration() override;
    void update_centroids() override;

    std::string get_name() const override {return "SOA Sequential SLIC";}
    DataLayout get_data_layout() const override {return DataLayout::SoA;}
    bool is_parallel() const override {return false;}

};
#endif //SLIC_SEGMENTATION_ALGORITHM_SLIC_ALGORITHM_SOA_SEQUENTIAL_H