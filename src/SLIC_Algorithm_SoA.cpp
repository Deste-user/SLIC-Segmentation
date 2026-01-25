#include "../include/SLIC_Algorithm_SoA.h"



void SLIC_Algorithm_SoA::clear() {

    for (int y = 0; y < image_lab.rows; y++) {
        for (int x = 0; x < image_lab.cols; x++) {
            int idx = y * image_lab.cols + x;
            cv::Vec3b lab_pixel = image_lab.at<cv::Vec3b>(y, x);

            img->L[idx] = (double)lab_pixel[0];
            img->A[idx] = (double)lab_pixel[1];
            img->B[idx] = (double)lab_pixel[2];
            img->x[idx] = x;
            img->y[idx] = y;
            img->distances[idx] = MAXFLOAT;
            img->labels[idx] = -1;
        }
    }

    for (int i = 0; i < K; i++) {
        super_pixels->label[i] = i;
        super_pixels->val_L[i] = 0.0f;
        super_pixels->val_a[i] = 0.0f;
        super_pixels->val_b[i] = 0.0f;
        super_pixels->centroid_x[i] = 0;
        super_pixels->centroid_y[i] = 0;
    }
}


int SLIC_Algorithm_SoA::EnforceConnectivity() {
        int num_pixels = this->N;
        int* new_labels = new int[num_pixels];
        for(int i = 0; i < num_pixels; i++) {
        new_labels[i] = -1;
        }

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

        for (int y = 0; y < this->image_lab.rows; y++) {
            for (int x = 0; x < this->image_lab.cols; x++) {
                int idx = y * this->image_lab.cols + x;

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
                    else if (y > 0 && new_labels[idx - this->image_lab.cols] >= 0) best_adj_label = new_labels[idx - this->image_lab.cols];

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
                            if (nx >= 0 && nx < this->image_lab.cols && ny >= 0 && ny < this->image_lab.rows) {
                                int n_idx = ny * this->image_lab.cols + nx;

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
                            int r_idx = y_vec[k] * this->image_lab.cols + x_vec[k];
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

        delete[] new_labels;
        return final_label_count;
}

cv::Mat SLIC_Algorithm_SoA::display_boundaries() {

        cv::Mat lab_mat(this->image_lab.rows, this->image_lab.cols, CV_8UC3);

        for (int y = 0; y < this->image_lab.rows; y++) {
            for (int x = 0; x < this->image_lab.cols; x++) {
                int idx = y * this->image_lab.cols + x;
                int label = img->labels[idx];

                if (label >= 0) {
                    lab_mat.at<cv::Vec3b>(y, x)[0] = (uchar)super_pixels->val_L[label];
                    lab_mat.at<cv::Vec3b>(y, x)[1] = (uchar)super_pixels->val_a[label];
                    lab_mat.at<cv::Vec3b>(y, x)[2] = (uchar)super_pixels->val_b[label];
                }
            }
        }

        cv::Mat output;
        cv::cvtColor(lab_mat, output, cv::COLOR_Lab2BGR);

        for (int y = 0; y < this->image_lab.rows - 1; y++) {
            for (int x = 0; x < this->image_lab.cols - 1; x++) {
                int idx = y * this->image_lab.cols + x;
                int idx_right = idx + 1;
                int idx_down = idx + this->image_lab.cols;

                if (img->labels[idx] != img->labels[idx_right] ||
                    img->labels[idx] != img->labels[idx_down]) {
                    output.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                    }
            }
        }

        for (int x = 0; x < this->image_lab.cols - 1; x++) {
            int idx = (this->image_lab.rows - 1) * this->image_lab.cols + x;
            if (img->labels[idx] != img->labels[idx + 1]) {
                output.at<cv::Vec3b>(this->image_lab.rows - 1, x) = cv::Vec3b(0, 0, 0);
            }
        }

        for (int y = 0; y < this->image_lab.rows - 1; y++) {
            int idx = y * this->image_lab.cols + (this->image_lab.cols - 1);
            if (img->labels[idx] != img->labels[idx + this->image_lab.cols]) {
                output.at<cv::Vec3b>(y, this->image_lab.cols - 1) = cv::Vec3b(0, 0, 0);
            }
        }

        cv::imshow("Segmentation",output);
        cv::waitKey(0);
        return output;
}

float SLIC_Algorithm_SoA::calculate_gradient(int x, int y) {
    float gradient = 0;

    if (x <= 0 || x >= this->image_lab.cols - 1 || y <= 0 || y >= this->image_lab.rows) return INT_MAX;

    int idx_right = y * this->image_lab.cols + (x + 1);
    int idx_left = y * this->image_lab.cols + (x - 1);
    int idx_down = (y + 1) * this->image_lab.cols + x;
    int idx_up = (y - 1) * this->image_lab.cols + x;

    // Differences L
    float diff_x_L = img->L[idx_right] - img->L[idx_left];
    float diff_y_L = img->L[idx_down] - img->L[idx_up];

    // Differences A
    float diff_x_A = img->A[idx_right] - img->A[idx_left];
    float diff_y_A = img->A[idx_down] - img->A[idx_up];

    // Differences B
    float diff_x_B = img->B[idx_right] - img->B[idx_left];
    float diff_y_B = img->B[idx_down] - img->B[idx_up];

    gradient = diff_x_L * diff_x_L + diff_y_L * diff_y_L +
               diff_x_A * diff_x_A + diff_y_A * diff_y_A +
               diff_x_B * diff_x_B + diff_y_B * diff_y_B;

    return gradient;
}
