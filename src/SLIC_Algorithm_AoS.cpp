#include "../include/SLIC_Algorithm_AoS.h"

void SLIC_Algorithm_AoS::clear(){

    for (int y = 0; y < this->image_lab.rows; y++) {
        for (int x = 0; x < this->image_lab.cols; x++) {
            int idx = y * this->image_lab.cols + x;
            cv::Vec3b lab_pixel = this->image_lab.at<cv::Vec3b>(y, x);

            this->pxls[idx].L = (float) lab_pixel[0];
            this->pxls[idx].A = (float) lab_pixel[1];
            this->pxls[idx].B = (float) lab_pixel[2];
            this->pxls[idx].x = x;
            this->pxls[idx].y = y;
            this->pxls[idx].label = -1;
            this->pxls[idx].distance= MAXFLOAT;
        }
    }

    // Inizializza i superpixel
    for (int i = 0; i < K; i++) {
        this->spxls[i].label = i;
        this->spxls[i].val_L = 0.0f;
        this->spxls[i].val_a = 0.0f;
        this->spxls[i].val_b = 0.0f;
        this->spxls[i].centroid_x = 0;
        this->spxls[i].centroid_y = 0;
    }
}

int SLIC_Algorithm_AoS::EnforceConnectivity(){
        int *new_labels = (int *) malloc(N * sizeof(int));
        for (int i = 0; i < N; i++) new_labels[i] = -1;

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


        for (int y = 0; y < this->image_lab.rows; y++) {
            for (int x = 0; x < this->image_lab.cols; x++) {
                int idx = y * this->image_lab.cols + x;

                // If pixel is not processed
                if (new_labels[idx] < 0) {
                    // Save the label we are currently working on
                    int current_label = pxls[idx].label;

                    // Prepare the BFS
                    x_vec.clear();
                    y_vec.clear();
                    x_vec.push_back(x);
                    y_vec.push_back(y);

                    // Assign the new label to the starting pixel
                    new_labels[idx] = final_label_count;

                    int count = 1;
                    int best_adj_label = -1;

                    // Find a valid neighbor label (locally)
                    if (x > 0 && new_labels[idx - 1] >= 0) {
                        best_adj_label = new_labels[idx - 1];
                    } else if (y > 0 && new_labels[idx - this->image_lab.cols] >= 0) {
                        best_adj_label = new_labels[idx - this->image_lab.cols];
                    }

                    // BFS
                    int vec_idx = 0;
                    while (vec_idx < x_vec.size()) {
                        // extract the current pixel
                        int cx = x_vec[vec_idx];
                        int cy = y_vec[vec_idx];
                        vec_idx++;
                        // Check the four neighbors
                        for (int d = 0; d < 4; d++) {
                            int nx = cx + dx[d];
                            int ny = cy + dy[d];

                            if (nx >= 0 && nx < this->image_lab.cols && ny >= 0 && ny < this->image_lab.rows) {
                                int n_idx = ny * this->image_lab.cols + nx;

                                // If it has the current label and is not yet labeled in the new map
                                if (new_labels[n_idx] < 0 && pxls[n_idx].label == current_label) {
                                    new_labels[n_idx] = final_label_count;
                                    x_vec.push_back(nx);
                                    y_vec.push_back(ny);
                                    count++;
                                }
                            }
                        }
                    }


                    if (count <= MIN_SIZE) {
                        // Small group - Join all
                        int target_label = (best_adj_label >= 0) ? best_adj_label : adj_label;

                        // Rename all this group
                        for (size_t k = 0; k < x_vec.size(); k++) {
                            int r_idx = y_vec[k] * this->image_lab.cols + x_vec[k];
                            new_labels[r_idx] = target_label;
                        }
                    } else {
                        // This became  new valid neighbor for next ones
                        adj_label = final_label_count;
                        final_label_count++;
                    }
                }
            }
        }

        // Copia finale
        for (int i = 0; i < N; i++) {
            pxls[i].label = new_labels[i];
        }

        free(new_labels);

        // Utile ritornare il numero reale di superpixel trovati
        return final_label_count;
}

float SLIC_Algorithm_AoS::calculate_gradient(int x, int y) {
    float gradient = 0.0f;

    if (x <= 0 || x >= this->image_lab.cols - 1 || y <= 0 || y >= this->image_lab.rows - 1) return FLT_MAX;
    int idx_right = y * this->image_lab.cols + (x + 1);
    int idx_left = y * this->image_lab.cols + (x - 1);
    int idx_down = (y + 1) * this->image_lab.cols + x;
    int idx_up = (y - 1) * this->image_lab.cols + x;

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

cv::Mat SLIC_Algorithm_AoS::display_boundaries() {
    cv::Mat lab_mat(this->image_lab.rows, this->image_lab.cols, CV_8UC3);
    for (int y = 0; y < this->image_lab.rows; y++) {
        for (int x = 0; x < this->image_lab.cols; x++) {
            int idx = y * this->image_lab.cols + x;
            lab_mat.at<cv::Vec3b>(y, x)[0] = (uchar) spxls[pxls[idx].label].val_L;
            lab_mat.at<cv::Vec3b>(y, x)[1] = (uchar) spxls[pxls[idx].label].val_a;
            lab_mat.at<cv::Vec3b>(y, x)[2] = (uchar) spxls[pxls[idx].label].val_b;
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
    cv::imshow("Segmentation",output_mat);
    cv::waitKey(0);

    return output_mat;
}
