#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

#include "FAST.h"

//
//std::vector<Corner> fastCorners(const cv::Mat& image, int threshold, int n) {
//    std::vector<Corner> corners;
//    static const int dx[16] = { 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3 };
//    static const int dy[16] = { 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1 };
//
//    for (int y = 3; y < image.rows - 3; y++) {
//        for (int x = 3; x < image.cols - 3; x++) {
//            uchar centerPixel = image.at<uchar>(y, x);
//            float sadScore = 0.0f;
//            int contiguousBrighter = 0, contiguousDarker = 0;
//
//            for (int i = 0; i < 16; i++) {
//                uchar candidatePixel = image.at<uchar>(y + dy[i], x + dx[i]);
//                int diff = static_cast<int>(candidatePixel) - static_cast<int>(centerPixel);
//                sadScore += std::abs(diff);
//
//                if (diff > threshold) {
//                    contiguousBrighter = (contiguousDarker > 0) ? 0 : contiguousBrighter + 1;
//                    contiguousDarker = 0;
//                }
//                else if (diff < -threshold) {
//                    contiguousDarker = (contiguousBrighter > 0) ? 0 : contiguousDarker + 1;
//                    contiguousBrighter = 0;
//                }
//                else {
//                    contiguousBrighter = contiguousDarker = 0;
//                }
//
//                if (contiguousBrighter >= n || contiguousDarker >= n) {
//                    corners.push_back({ cv::Point(x, y), sadScore });
//                    break;
//                }
//            }
//        }
//    }
//
//    return corners;
//}
//
//
//bool compareCorners(const Corner& a, const Corner& b) {
//    return a.score > b.score; // Descending order
//}
//
//std::vector<cv::Point> nonMaximumSuppression(const std::vector<Corner>& corners, float threshold) {
//    std::vector<Corner> sortedCorners = corners;
//    std::sort(sortedCorners.begin(), sortedCorners.end(), compareCorners);
//
//    std::vector<cv::Point> result;
//    std::vector<bool> suppressed(sortedCorners.size(), false);
//
//    for (size_t i = 0; i < sortedCorners.size(); i++) {
//        if (!suppressed[i]) {
//            result.push_back(sortedCorners[i].position);
//
//            for (size_t j = i + 1; j < sortedCorners.size(); j++) {
//                if (!suppressed[j]) {
//                    double dist = cv::norm(sortedCorners[i].position - sortedCorners[j].position);
//                    if (dist < threshold) {
//                        suppressed[j] = true;
//                    }
//                }
//            }
//        }
//    }
//
//    return result;
//}







#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>

#include "FAST.h"



// Assuming a maximum of NMAX features.
//const int NMAX = 100; // Adjust this number based on your requirements

void fastCorners(const cv::Mat& image, int threshold, int n, featurePoint features[NMAX]) {
    int featureCount = 0;
    static const int dx[16] = { 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3 };
    static const int dy[16] = { 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1 };
    int sadMax = 0;
    for (int y = 3; y < image.rows - 3; y++) {
        for (int x = 3; x < image.cols - 3; x++) {
            uchar centerPixel = image.at<uchar>(y, x);
            int sadScore = 0;
            int contiguousBrighter = 0, contiguousDarker = 0;

            for (int i = 0; i < 16; i++) {
                uchar candidatePixel = image.at<uchar>(y + dy[i], x + dx[i]);
                int diff = static_cast<int>(candidatePixel) - static_cast<int>(centerPixel);
                sadScore += std::abs(diff);

                if (diff > threshold) {
                    contiguousBrighter = (contiguousDarker > 0) ? 0 : contiguousBrighter + 1;
                    contiguousDarker = 0;
                }
                else if (diff < -threshold) {
                    contiguousDarker = (contiguousBrighter > 0) ? 0 : contiguousDarker + 1;
                    contiguousBrighter = 0;
                }
                else {
                    contiguousBrighter = contiguousDarker = 0;
                }

                if (contiguousBrighter >= n || contiguousDarker >= n) {
                    if (featureCount < NMAX && sadScore > 100) {
                        features[featureCount++] = { x, y, sadScore, true };
                        if (sadScore > sadMax)
                            sadMax = sadScore;
                    }
                    break;
                }
            }
        }
    }
    cout << endl << "sadMax: " << sadMax << endl;
    // Set remaining features as inactive
    for (int i = featureCount; i < NMAX; i++) {
        features[i].active = false;
    }
}

bool compareFeaturePoints(const featurePoint& a, const featurePoint& b) {
    return a.score > b.score;
}

void nonMaximumSuppression(featurePoint features[NMAX], float threshold) {
    std::sort(features, features + NMAX, compareFeaturePoints);

    for (int i = 0; i < NMAX; i++) {
        if (!features[i].active) continue;

        for (int j = i + 1; j < NMAX; j++) {
            if (!features[j].active) continue;

            double dist = std::hypot(features[i].x - features[j].x, features[i].y - features[j].y);

            if (dist < threshold) {
                features[j].active = false;
            }
        }
    }
}

int main1() {
    // Load image in grayscale
    cv::Mat image = cv::imread("C:/Users/Mngzr/Desktop/BILLEDANAL WLLIAM/Imageanalysis_on_uC_30330-main/stereo_video_v2/left/l01.png", cv::IMREAD_GRAYSCALE);

    // Corner detection
    int threshold = 30; // Define a suitable threshold
    int n = 9; //number of contigous pixels
    // std::vector<Corner> corners = fastCorners(image, threshold,9);
    featurePoint list[NMAX];
    fastCorners(image, threshold, 9, list);

   // nonMaximumSuppression(list, 10);
    //std::vector<cv::Point> corners_suppressed = nonMaximumSuppression(fastCorners(image, threshold, 9), 10); // minimum distance
    // Draw corners on the image
    cv::cvtColor(image, image, COLOR_GRAY2BGR);
    for (const auto& corner : list) {
        if(corner.active)
        cv::circle(image, cv::Point(corner.x,corner.y), 8, cv::Scalar(0, 0, 255), 2);
    }
    nonMaximumSuppression(list, 20);
    for (const auto& corner : list) {
        if (corner.active)
            cv::circle(image, cv::Point(corner.x, corner.y), 8, cv::Scalar(0, 255, 0), 2);
    }
    // Display the result
    cv::imshow("Corners", image);
    cv::waitKey(0);

    return 0;
}
