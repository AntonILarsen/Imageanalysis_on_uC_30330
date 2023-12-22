#ifndef FASTCORNERDETECTOR_H
#define FASTCORNERDETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include "FeatureDetection.h"


// Function declarations

void fastCorners(const cv::Mat& image, int threshold, int n, featurePoint features[NMAX]);
bool compareFeaturePoints(const featurePoint& a, const featurePoint& b);
void nonMaximumSuppression(featurePoint features[NMAX], float threshold);
#endif // FASTCORNERDETECTOR_H
