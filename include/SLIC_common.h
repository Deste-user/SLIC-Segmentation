#ifndef SLIC_SEGMENTATION_ALGORITHM_SLIC_COMMON_H
#define SLIC_SEGMENTATION_ALGORITHM_SLIC_COMMON_H

#include <opencv2/opencv.hpp>
#include <string>
const int K = 200;
// Utilità condivise
std::string get_random_image_path(const std::string& folder_path);
double distance_SLIC(float cL, float cA, float cB, int cx, int cy,
                     float pL, float pA, float pB, int px, int py, int S, int m);





#endif //SLIC_SEGMENTATION_ALGORITHM_SLIC_COMMON_H