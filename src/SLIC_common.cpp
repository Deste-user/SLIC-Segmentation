#include "slic_common.h"
#include <filesystem>
#include <vector>
#include <random>
#include <algorithm>

namespace fs = std::filesystem;


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


double distance_SLIC(float cL, float cA, float cB, int cx, int cy,
                     float pL, float pA, float pB, int px, int py, int S, int m) {
    double d_color = std::sqrt(
        (cL - pL) * (cL - pL) +
        (cA - pA) * (cA - pA) +
        (cB - pB) * (cB - pB)
    );

    double d_spatial = std::sqrt(
        (cx - px) * (cx - px) +
        (cy - py) * (cy - py)
    );

    return std::sqrt(d_color * d_color + (d_spatial / S * m) * (d_spatial / S * m));
}