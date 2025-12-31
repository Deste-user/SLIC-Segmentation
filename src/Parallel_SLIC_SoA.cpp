// In this file we implement the parallel version of SLIC using Structure of Arrays (SoA) approach.
#include "SLIC_common.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <random>
#include <omp.h>


// Impossibile per Legge di Amhdal -> misurare la tempistica.
// Considerare la connessione anche negli altri casi.


struct SuperPixels {
    int* label;
    int *centroid_x;
    int *centroid_y;
    float* val_L;
    float* val_a;
    float* val_b;
};

struct Image {
    float *L, *A, *B, *distances;
    int *x,*y, *labels;
};


class SLIC_Algorithm_SoA_Parallel : public SLIC_Algorithm {
    public:
    std::string get_name() const override {return "SOA Parallel SLIC";}
    DataLayout get_data_layout() const override {return DataLayout::SoA;}
    bool is_parallel() const override {return true;}

    void Initialization_Structures(Image* img, SuperPixels* super_pixels, cv::Mat image_lab, int K) {
        int N = image_lab.cols * image_lab.rows;

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

        // PARALLELIZZAZIONE OTTIMIZZATA
        // collapse(2): fonde y e x in un unico ciclo da 0 a N
        // Non serve 'private(idx)' se dichiariamo le variabili dentro
        #pragma omp parallel for collapse(2) schedule(static)
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

        // Initialization super pixels
        super_pixels->centroid_x = (int*)malloc(K * sizeof(int));
        super_pixels->centroid_y = (int*)malloc(K * sizeof(int));
        super_pixels->val_L = (float*)malloc(K * sizeof(float));
        super_pixels->val_a = (float*)malloc(K * sizeof(float));
        super_pixels->val_b = (float*)malloc(K * sizeof(float));
        super_pixels->label = (int*)malloc(K * sizeof(int));

        // Questo ciclo è troppo piccolo per parallelizzarlo (overhead > guadagno).
        for (int l = 0; l < K; l++) {
            super_pixels->label[l] = l;
        }
    }

    int calculate_gradient(Image* img, int x, int y, const int cols,const int rows) {
    int gradient = 0;

    if (x <= 0 || x >= cols - 1 || y <= 0 || y >= rows) return INT_MAX;

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

    int EnforceConnectivity(Image* img, const int rows, const int cols, int K) {
        int num_pixels = rows * cols;

        // Nuova matrice per le label "pulite"
        // Inizializzata a -1 (non visitato)
        int* new_labels = (int*)malloc(num_pixels * sizeof(int));
        for(int i=0; i<num_pixels; i++) new_labels[i] = -1;

        // Dimensione media attesa di un superpixel
        int superpixel_size = num_pixels / K;
        // Soglia: se un pezzo è più piccolo di questo, viene assorbito
        int threshold = superpixel_size >> 2; // Equivalente a size / 4 (25%)

        // Array per muoversi nei 4 vicini (dx, dy)
        const int dx[] = {1, -1, 0, 0};
        const int dy[] = {0, 0, 1, -1};

        // Vettori per la ricerca (x e y)
        // Usiamo std::vector come stack per evitare ricorsione
        std::vector<int> x_vec;
        std::vector<int> y_vec;
        x_vec.reserve(superpixel_size); // Pre-allocazione per velocità
        y_vec.reserve(superpixel_size);

        int adj_label = 0; // Label del vicino a cui unirsi
        int final_label_count = 0; // Contatore per le nuove label sequenziali

        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                int idx = y * cols + x;

                // Se il pixel non è ancora stato processato nella nuova mappa
                if (new_labels[idx] < 0) {

                    // Salviamo la label originale che stiamo tracciando
                    int current_label = img->labels[idx];

                    // Iniziamo la ricerca di componenti connesse
                    x_vec.clear();
                    y_vec.clear();

                    // Aggiungiamo il primo punto
                    x_vec.push_back(x);
                    y_vec.push_back(y);
                    new_labels[idx] = final_label_count; // Assegnazione temporanea

                    int count = 1; // Quanti pixel in questo gruppo?

                    // Troviamo un'etichetta adiacente valida nel caso serva un merge.
                    // Cerchiamo nei vicini già visitati (es. sopra o sinistra)
                    int best_adj_label = -1;

                    // Controllo rapido vicini (sinistra e sopra) per trovare un "genitore"
                    if (x > 0 && new_labels[idx - 1] >= 0) best_adj_label = new_labels[idx - 1];
                    else if (y > 0 && new_labels[idx - cols] >= 0) best_adj_label = new_labels[idx - cols];

                    // --- INIZIO BFS/DFS ---
                    int vec_idx = 0;
                    while(vec_idx < x_vec.size()){
                        int cx = x_vec[vec_idx];
                        int cy = y_vec[vec_idx];
                        vec_idx++;

                        // Controlla i 4 vicini
                        for (int d = 0; d < 4; d++) {
                            // Spostamento con array definiti prima
                            int nx = cx + dx[d];
                            int ny = cy + dy[d];

                            // Controllo confini
                            if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
                                int n_idx = ny * cols + nx;

                                // Se ha label originale non ancora visitata
                                if (new_labels[n_idx] < 0 && img->labels[n_idx] == current_label) {
                                    new_labels[n_idx] = final_label_count; // Marca come visitato
                                    x_vec.push_back(nx);
                                    y_vec.push_back(ny);
                                    count++;
                                }
                            }
                        }
                    }
                    // --- FINE BFS ---

                    // DECISIONE: Tenere o Unire?
                    if (count <= threshold) {
                        // TROPPO PICCOLO -> UNISCI (MERGE)
                        // Se non abbiamo trovato un vicino prima, proviamo a usare adj_label globale
                        // (o l'ultimo valido). Se proprio non c'è (primo blocco), non facciamo nulla.
                        int target_label = (best_adj_label >= 0) ? best_adj_label : adj_label;

                        for (size_t k = 0; k < x_vec.size(); k++) {
                            int r_idx = y_vec[k] * cols + x_vec[k];
                            new_labels[r_idx] = target_label;
                        }
                        // Non incrementiamo final_label_count perché questo gruppo è sparito
                    } else {
                        // ABBASTANZA GRANDE -> MANTIENI
                        // È un nuovo superpixel valido.
                        adj_label = final_label_count; // Diventa il "vicino valido" per i prossimi
                        final_label_count++;
                    }
                }
            }
        }

        // Sovrascriviamo le label originali con quelle pulite
        // IMPORTANTE: il numero di superpixel reali (final_label_count)
        // potrebbe essere minore di K originale.
        for (int i = 0; i < num_pixels; i++) {
            img->labels[i] = new_labels[i];
        }

        // Aggiorna K reale se necessario, oppure lascia K invariato
        // (ma sapendo che alcune label > final_label_count sono vuote).

        free(new_labels);
        return final_label_count;
    }

    void Initialization(Image* img,SuperPixels* sp ,int S, const int rows, const int cols, int K) {

        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = S/2 ; y < rows; y += S) {
            for (int x = S/2 ; x < cols; x += S) {
                int idx = x + cols*y;
                int i = (y / S) * (cols / S) + (x / S);
                if (i >= K) continue;
                sp->centroid_x[i] = x;
                sp->centroid_y[i] = y;
                sp->val_L[i] = img->L[idx];
                sp->val_a[i] = img->A[idx];
                sp->val_b[i] = img->B[idx];
            }
        }

        // Spostamento su gradiente minimo (3x3)
        #pragma omp parallel for schedule(static)
        for (int k=0 ; k < K; k++) {
            int min_gradient = INT_MAX;
            int best_x = sp->centroid_x[k];
            int best_y = sp->centroid_y[k];

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int ny = sp->centroid_y[k] + dy;
                    int nx = sp->centroid_x[k] + dx;
                    if (nx > 0 && nx < cols - 1 && ny > 0 && ny < rows - 1) {
                        int g = calculate_gradient(img, nx, ny, cols,rows);
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
            int idx = best_y*cols+best_x;
            sp->val_L[k] = img->L[idx];
            sp->val_a[k] = img->A[idx];
            sp->val_b[k] = img->B[idx];
        }
    }

    // Use the pixel centric version - each pixel checks all superpixels in its 2Sx2S regio
    void iteration(Image* img, SuperPixels* sp, int S, const int rows, const int cols, int K, int m){
        if (!this->use_tiling) {
#pragma omp parallel for collapse(2) schedule(static)
            for (int y=0;y<rows;y++) {
                for (int x=0;x<cols;x++) {
                    int idx = x + cols*y;

                    int grid_x = x / S;
                    int grid_y = y / S;

                    for ( int ny= -1; ny <= 1; ny++) {
                        for (int nx=-1; nx <= 1;nx++) {
                            int kx = grid_x + nx;
                            int ky = grid_y + ny;
                            int k = ky * (cols / S) + kx;
                            if (k < 0 || k >= K) continue;
                            if (abs(sp->centroid_x[k] - x) < 2 * S &&
                                    abs(sp->centroid_y[k] - y) < 2 * S) {
                                double d = distance_SLIC(sp->val_L[k],sp->val_a[k],sp->val_b[k],sp->centroid_x[k], sp->centroid_y[k],
                                img->L[idx],img->A[idx], img->B[idx], img->x[idx], img->y[idx],S,m);

                                if (d < img->distances[idx]) {
                                    img->distances[idx] = d;
                                    img->labels[idx] = k;
                                }
                                    }
                        }
                    }
                }
            }
        }else{
#pragma omp parallel for collapse(2) schedule(static)
            // quando accedo ad una locazione, carico sulla cache un blocco di dati.
            // Di conseguenza accedo a blocchi di dati contigui per sfruttare la cache al meglio.
            for (int by=0; by<rows; by += TILE_SIZE) {
                for (int bx=0; bx<cols; bx += TILE_SIZE) {
                    int y_end = std::min(by + TILE_SIZE, rows);
                    int x_end = std::min(bx + TILE_SIZE, cols);
                    // Sfrutto la cache.
                    for (int y=by; y<y_end; y++) {
                        for (int x=bx; x<x_end; x++) {
                            int idx = x + cols*y;

                            int grid_x = x / S;
                            int grid_y = y / S;

                            for ( int ny= -1; ny <= 1; ny++) {
                                for (int nx=-1; nx <= 1;nx++) {
                                    int kx = grid_x + nx;
                                    int ky = grid_y + ny;
                                    int k = ky * (cols / S) + kx;
                                    if (k < 0 || k >= K) continue;
                                    if (abs(sp->centroid_x[k] - x) < 2 * S &&
                                            abs(sp->centroid_y[k] - y) < 2 * S) {
                                        double d = distance_SLIC(sp->val_L[k],sp->val_a[k],sp->val_b[k],sp->centroid_x[k], sp->centroid_y[k],
                                                                 img->L[idx],img->A[idx], img->B[idx], img->x[idx], img->y[idx],S,m);

                                        if (d < img->distances[idx]) {
                                            img->distances[idx] = d;
                                            img->labels[idx] = k;
                                        }
                                            }
                                }
                            }
                        }
                    }
                }

            }
        }
    }

    void update_centroids(Image* img, SuperPixels* sp, const int rows, const int cols, int K) {
        // Array per accumulatori
        double* global_sum_x = (double*)calloc(K, sizeof(double));
        double* global_sum_y = (double*)calloc(K, sizeof(double));
        double* global_sum_L = (double*)calloc(K, sizeof(double));
        double* global_sum_a = (double*)calloc(K, sizeof(double));
        double* global_sum_b = (double*)calloc(K, sizeof(double));
        int* global_count = (int*)calloc(K, sizeof(int));
        const int N = rows*cols;

        // Se faccio la parallelizzazione qui, si può verificare race conditions
    #pragma omp parallel
        {
            double* local_sum_x = (double*)calloc(K, sizeof(double));
            double* local_sum_y = (double*)calloc(K, sizeof(double));
            double* local_sum_L = (double*)calloc(K, sizeof(double));
            double* local_sum_a = (double*)calloc(K, sizeof(double));
            double* local_sum_b = (double*)calloc(K, sizeof(double));
            int* local_count = (int*)calloc(K, sizeof(int));

    #pragma omp for schedule(static) nowait
            // si usa nowait poichè non serve aspettare gli altri.
            for (int idx = 0; idx < N; idx++) {
                int lbl = img->labels[idx];
                if (lbl >= 0 && lbl < K) {
                    local_sum_L[lbl] += img->L[idx];
                    local_sum_a[lbl] += img->A[idx];
                    local_sum_b[lbl] += img->B[idx];
                    local_sum_x[lbl] += img->x[idx];
                    local_sum_y[lbl] += img->y[idx];
                    local_count[lbl]++;
                }
            }
            // Riduzione dei risultati locali nelle strutture globali.
    #pragma omp critical
            {
                for (int k = 0; k < K; k++) {
                    global_sum_x[k] += local_sum_x[k];
                    global_sum_y[k] += local_sum_y[k];
                    global_sum_L[k] += local_sum_L[k];
                    global_sum_a[k] += local_sum_a[k];
                    global_sum_b[k] += local_sum_b[k];
                    global_count[k] += local_count[k];
                }
            }
            // Libera memoria locale di ogni thread
            free(local_sum_x);
            free(local_sum_y);
            free(local_sum_L);
            free(local_sum_a);
            free(local_sum_b);
            free(local_count);
        }

        #pragma omp parallel for schedule(static)
        for (int k=0; k < K; k++) {
            if (global_count[k] > 0) {
                sp->centroid_x[k] = (int)(global_sum_x[k] / global_count[k]);
                sp->centroid_y[k] = (int)(global_sum_y[k] / global_count[k]);
                sp->val_L[k] = (float)(global_sum_L[k] / global_count[k]);
                sp->val_a[k] = (float)(global_sum_a[k] / global_count[k]);
                sp->val_b[k] = (float)(global_sum_b[k] / global_count[k]);
            }
        }

        // Libera memoria globale
        free(global_sum_x);
        free(global_sum_y);
        free(global_sum_L);
        free(global_sum_a);
        free(global_sum_b);
        free(global_count);
    }

    // Non parallelizzo questa funzione. Mi serve solo per visualizzare meglio l'output
    cv::Mat display_boundaries(Image* img, SuperPixels* sp, int rows, int cols) {
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

    void run(Image* img, SuperPixels* sp,int S, const int row, const int cols, int K,int num_iterations = 10) {
        Initialization(img, sp, S, row, cols, K);
        for (int i = 0; i < num_iterations; i++) {
            iteration(img,sp, S, row, cols, K, 10);
            update_centroids(img, sp, row, cols, K);
        }
        int real_num_labels = EnforceConnectivity(img, row, cols, K);
        update_centroids(img, sp, row, cols, real_num_labels);
    }
};
