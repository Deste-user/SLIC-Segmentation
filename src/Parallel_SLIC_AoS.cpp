#include <opencv2/opencv.hpp>
#include "SLIC_common.h"
#include <omp.h>

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

class SLIC_Algorithm_Parallel_AoS: public SLIC_Algorithm {
private:
    Pixel* pxls;
    SuperPixel* spxls;
    cv::Mat image_lab;
    int K;
    int N;
    int S;
    int m;
    int num_iterations;

public:
    std::string get_name() const override {return "AoS Parallel SLIC";}
    DataLayout get_data_layout() const override {return DataLayout::AoS;}
    bool is_parallel() const override {return true;}
    float calculate_gradient(int x, int y) override {
        float gradient = 0.0f;

        if (x<=0 || x >= this->image_lab.cols -1 || y <=0 || y >= image_lab.rows -1) return FLT_MAX;

        int idx_right = y * image_lab.cols + (x + 1);
        int idx_left = y * image_lab.cols + (x - 1);
        int idx_down = (y + 1) * image_lab.rows + x;
        int idx_up = (y - 1) * image_lab.rows + x;

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
    SLIC_Algorithm_Parallel_AoS(cv::Mat image_lab, int K, int m, int iterations) {
        this->N= image_lab.cols * image_lab.rows;
        this->K= K;
        this->m= m;
        this->image_lab= image_lab;
        this->num_iterations= iterations;
        pxls= (Pixel*) malloc(N * sizeof(Pixel));
        this->S = (int)std::sqrt((double)(image_lab.rows*image_lab.cols) / K);
        int cols_steps = image_lab.cols / S;
        int rows_steps = image_lab.rows / S;
        this->K= rows_steps * cols_steps;
        spxls= (SuperPixel*) malloc(this->K * sizeof(SuperPixel));

        // Linearization of the Image.
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < image_lab.rows; y++) {
            for (int x = 0; x < image_lab.cols; x++) {
                int idx= y * image_lab.cols + x;
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
        #pragma omp parallel for schedule(static)
        for (int i=0; i < K; i++) {
            spxls[i].label = i;
        }
    }
    ~SLIC_Algorithm_Parallel_AoS() {
        free(pxls);
        free(spxls);
    }
    void Initialization() override{
        // To initialize the centroids of superpixels in a grid pattern
        #pragma omp parallel for collapse(2) schedule(static)
        for (int y = S/2 ; y < this->image_lab.rows; y += S) {
            for (int x = S/2 ; x < this->image_lab.cols; x += S) {
                int idx= x + this->image_lab.cols*y;
                int i = (y / S) * (this->image_lab.cols / S) + (x / S);
                if (i >= K) continue;
                spxls[i].centroid_x = x;
                spxls[i].centroid_y = y;
                spxls[i].val_L = pxls[idx].L;
                spxls[i].val_a = pxls[idx].A;
                spxls[i].val_b = pxls[idx].B;
            }
        }

        // To adjust centroids to the lowest gradient position in a 3x3 neighborhood
        #pragma omp parallel for schedule(static)
        for (int k=0; k<this->K; k++) {
            float min_gradient= FLT_MAX;
            int best_x= spxls[k].centroid_x;
            int best_y= spxls[k].centroid_y;
            for (int dy=-1; dy< 1; dy++) {
                for (int dx=-1; dx<1;dx++) {
                    int ny= spxls[k].centroid_y + S*dy;
                    int nx= spxls[k].centroid_x + S*dx;
                    if (nx > 0 && nx < this->image_lab.cols -1 && ny > 0 && ny < this->image_lab.rows -1) {
                        float g= calculate_gradient(nx, ny);
                        if (g < min_gradient) {
                            min_gradient= g;
                            best_x= nx;
                            best_y= ny;
                        }
                    }
                }
            }
            spxls[k].centroid_x= best_x;
            spxls[k].centroid_y= best_y;
        }
    }

    void iteration() override {
        if (!this->use_tiling) {
#pragma omp parallel for schedule(static)
            for (int y = 0; y < this->image_lab.rows; y++) {
                for (int x = 0; x < this->image_lab.cols; x++) {
                    int idx = x + this->image_lab.cols * y;

                    int grid_x = x / S;
                    int grid_y = y / S;

                    for (int ny = -1; ny <= 1; ny++) {
                        for (int nx = -1; nx <= 1; nx++) {
                            int kx = grid_x + nx;
                            int ky = grid_y + ny;
                            int k = ky * (this->image_lab.cols / S) + kx;
                            if (k < 0 || k >= K) continue;
                            if (abs(spxls[k].centroid_x - x) < 2 * S &&
                                abs(spxls[k].centroid_y - y) < 2 * S) {
                                double d = distance_SLIC(spxls[k].val_L, spxls[k].val_a, spxls[k].val_b,
                                                         spxls[k].centroid_x, spxls[k].centroid_y,
                                                         pxls[idx].L, pxls[idx].A, pxls[idx].B, pxls[idx].x,
                                                         pxls[idx].y, S, m);

                                if (d < pxls[idx].distance) {
                                    pxls[idx].distance = d;
                                    pxls[idx].label = k;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // 1. Calcola le dimensioni della griglia UNA VOLTA sola
            // Questo garantisce che l'indice 'k' sia coerente con la creazione dei superpixel
            int grid_width = this->image_lab.cols / S;
            int grid_height = this->image_lab.rows / S;

#pragma omp parallel for collapse(2) schedule(static)
            for (int ty = 0; ty < this->image_lab.rows; ty += TILE_SIZE) {
                for (int tx = 0; tx < image_lab.cols; tx += TILE_SIZE) {
                    int y_end = std::min(ty + TILE_SIZE, this->image_lab.rows);
                    int x_end = std::min(tx + TILE_SIZE, this->image_lab.cols);

                    for (int y = ty; y < y_end; y++) {
                        for (int x = tx; x < x_end; x++) {
                            int idx = x + this->image_lab.cols * y;

                            // 2. Calcola la cella della griglia "pura"
                            // SENZA "if (grid_x >= ...)" -> Lascialo sbordare!
                            int grid_x = x / S;
                            int grid_y = y / S;

                            // 3. Cerca nei 9 vicini
                            for (int ny = -1; ny <= 1; ny++) {
                                for (int nx = -1; nx <= 1; nx++) {
                                    int kx = grid_x + nx;
                                    int ky = grid_y + ny;

                                    // 4. Controllo che il CENTROIDE (kx, ky) esista davvero
                                    if (kx >= 0 && kx < grid_width &&
                                        ky >= 0 && ky < grid_height) {
                                        int k = ky * grid_width + kx;

                                        // Safety check
                                        if (k >= 0 && k < K) {
                                            // Controllo rapido spaziale (2S x 2S)
                                            // Se il pixel è troppo lontano, abs() lo scarta subito
                                            if (abs(spxls[k].centroid_x - x) < 2 * S &&
                                                abs(spxls[k].centroid_y - y) < 2 * S) {
                                                double d = distance_SLIC(
                                                    spxls[k].val_L, spxls[k].val_a, spxls[k].val_b, spxls[k].centroid_x,
                                                    spxls[k].centroid_y,
                                                    pxls[idx].L, pxls[idx].A, pxls[idx].B, pxls[idx].x, pxls[idx].y, S,
                                                    m
                                                );

                                                if (d < pxls[idx].distance) {
                                                    pxls[idx].distance = d;
                                                    pxls[idx].label = k;
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
        }
    }

    void update_centroids() override {
        double *global_sum_x = (double *) calloc(K, sizeof(double));
        double *global_sum_y = (double *) calloc(K, sizeof(double));
        double *global_sum_L = (double *) calloc(K, sizeof(double));
        double *global_sum_a = (double *) calloc(K, sizeof(double));
        double *global_sum_b = (double *) calloc(K, sizeof(double));
        int *global_count = (int *) calloc(K, sizeof(int));

#pragma omp parallel
        {
            double *local_sum_x = (double *) calloc(K, sizeof(double));
            double *local_sum_y = (double *) calloc(K, sizeof(double));
            double *local_sum_L = (double *) calloc(K, sizeof(double));
            double *local_sum_a = (double *) calloc(K, sizeof(double));
            double *local_sum_b = (double *) calloc(K, sizeof(double));
            int *local_count = (int *) calloc(K, sizeof(int));

#pragma omp for schedule(static) nowait
            for (int idx = 0; idx < N; idx++) {
                int lbl = pxls[idx].label;
                if (lbl >= 0 && lbl < K) {
                    local_sum_L[lbl] += pxls[idx].L;
                    local_sum_a[lbl] += pxls[idx].A;
                    local_sum_b[lbl] += pxls[idx].B;
                    local_sum_x[lbl] += pxls[idx].x;
                    local_sum_y[lbl] += pxls[idx].y;
                    local_count[lbl]++;
                }
            }

#pragma omp critical
            {
#pragma omp simd
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
        for (int k = 0; k < K; k++) {
            if (global_count[k] > 0) {
                spxls[k].centroid_x = (int) (global_sum_x[k] / global_count[k]);
                spxls[k].centroid_y = (int) (global_sum_y[k] / global_count[k]);
                spxls[k].val_L = (float) (global_sum_L[k] / global_count[k]);
                spxls[k].val_a = (float) (global_sum_a[k] / global_count[k]);
                spxls[k].val_b = (float) (global_sum_b[k] / global_count[k]);
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
    int EnforceConnectivity() override{

    // 1. Inizializzazione Array Label Pulite
    int* new_labels = (int*)malloc(N * sizeof(int));
    for(int i=0; i<N; i++) new_labels[i] = -1;

    // Parametri
    const int dx[] = {1, -1, 0, 0};
    const int dy[] = {0, 0, 1, -1};
    const int SUPPIXEL_SIZE = N / K;
    const int MIN_SIZE = SUPPIXEL_SIZE >> 2;

    int final_label_count = 0;
    int adj_label = 0; // Label di fallback per unire i piccoli pezzi

    std::vector<int> x_vec;
    std::vector<int> y_vec;
    x_vec.reserve(SUPPIXEL_SIZE * 4);
    y_vec.reserve(SUPPIXEL_SIZE * 4);

    // 2. Scorro l'immagine (PRIMA Y POI X per performance!)
    for (int y = 0; y < this->image_lab.rows; y++) {
        for (int x = 0; x < this->image_lab.cols; x++) {
            int idx = y * this->image_lab.cols + x;

            // Se il pixel non è ancora stato processato
            if (new_labels[idx] < 0) {

                // Salviamo la label che stiamo tracciando
                int current_label = pxls[idx].label;

                // Prepariamo la BFS
                x_vec.clear();
                y_vec.clear();
                x_vec.push_back(x);
                y_vec.push_back(y);

                // Assegniamo provvisoriamente la nuova label corrente
                new_labels[idx] = final_label_count;

                int count = 1;
                int best_adj_label = -1;

                // CORREZIONE 4: Trova un vicino valido (Sinistra o Sopra)
                // idx-1 è il pixel a sinistra, idx-cols è quello sopra
                if (x > 0 && new_labels[idx - 1] >= 0) {
                    best_adj_label = new_labels[idx - 1];
                } else if (y > 0 && new_labels[idx - this->image_lab.cols] >= 0) {
                    best_adj_label = new_labels[idx - this->image_lab.cols];
                }

                // --- INIZIO BFS ---
                int vec_idx = 0;
                while(vec_idx < x_vec.size()){
                    // Si estrae il pixel corrente
                    int cx = x_vec[vec_idx];
                    int cy = y_vec[vec_idx];
                    vec_idx++;
                    // Controlla i 4 vicini
                    for (int d = 0; d < 4; d++) {
                        int nx = cx + dx[d];
                        int ny = cy + dy[d];

                        if (nx >= 0 && nx < this->image_lab.cols && ny >= 0 && ny < this->image_lab.rows) {
                            int n_idx = ny * this->image_lab.cols + nx;

                            // Se ha la stessa label originale e non è ancora stato visitato
                            if (new_labels[n_idx] < 0 && pxls[n_idx].label == current_label) {
                                new_labels[n_idx] = final_label_count;
                                x_vec.push_back(nx);
                                y_vec.push_back(ny);
                                count++;
                            }
                        }
                    }
                }
                // --- FINE BFS ---

                if (count <= MIN_SIZE) {
                    // TROPPO PICCOLO -> UNISCI AL VICINO
                    // Se non abbiamo un vicino locale, usiamo l'ultimo valido globale
                    int target_label = (best_adj_label >= 0) ? best_adj_label : adj_label;

                    // Rinomina tutti i pixel di questo piccolo gruppo
                    for (size_t k = 0; k < x_vec.size(); k++) {
                        int r_idx = y_vec[k] * this->image_lab.cols + x_vec[k];
                        new_labels[r_idx] = target_label;
                    }
                } else {
                    // ABBASTANZA GRANDE -> MANTIENI
                    // Questo diventa il nuovo "vicino valido" per i prossimi
                    adj_label = final_label_count;
                    final_label_count++;
                }
            }
        }
    }

    // Copia finale
    for(int i=0; i<N; i++) {
        pxls[i].label = new_labels[i];
    }

    free(new_labels);

    // Utile ritornare il numero reale di superpixel trovati
    return final_label_count;

    }


    void run() override {
        Initialization();
        for (int i = 0; i < num_iterations; i++) {
            iteration();
            update_centroids();
        }
        this->K = EnforceConnectivity();
        update_centroids();
    }

    cv::Mat display_boundaries() override {
        cv::Mat lab_mat(this->image_lab.rows, this->image_lab.cols, CV_8UC3);
        for (int y = 0; y < this->image_lab.rows ; y++) {
            for (int x = 0; x < this->image_lab.cols ; x++) {
                int idx = y * this->image_lab.cols + x;
                lab_mat.at<cv::Vec3b>(y, x)[0] = (uchar)spxls[pxls[idx].label].val_L;
                lab_mat.at<cv::Vec3b>(y, x)[1] = (uchar)spxls[pxls[idx].label].val_a;
                lab_mat.at<cv::Vec3b>(y, x)[2] = (uchar)spxls[pxls[idx].label].val_b;
            }
        }
        cv::Mat output_mat;
        cv::cvtColor(lab_mat, output_mat, cv::COLOR_Lab2BGR);

        // Draw boundaries
        for (int y = 0; y < this->image_lab.rows - 1; y++) {
            for (int x = 0; x < this->image_lab.cols - 1; x++) {
                int idx = y * this->image_lab.cols + x;
                int idx_right = y * this->image_lab.cols + (x + 1);
                int idx_down = (y + 1) * this->image_lab.cols + x;

                if (pxls[idx].label != pxls[idx_right].label || pxls[idx].label != pxls[idx_down].label) {
                    output_mat.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0); // Black boundary
                }
            }
        }

        return output_mat;

    };
};


int main(){
    std::string img_path = get_random_image_path(PATH_images);

    cv::Mat image = cv::imread(img_path);
    cv::imshow("Original Image", image);

    if (image.empty()) return -1;

    cv::Mat image_lab;
    cv::cvtColor(image, image_lab, cv::COLOR_BGR2Lab);
    SLIC_Algorithm_Parallel_AoS slic_elab =SLIC_Algorithm_Parallel_AoS(image_lab, 600, 10,10);
    slic_elab.run();
    cv::Mat output= slic_elab.display_boundaries();
    cv::imshow("SLIC Result Parallel AoS", output);

    cv::waitKey(0);

}