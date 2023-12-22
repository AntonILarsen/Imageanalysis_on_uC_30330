#pragma once

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <filesystem>
#include <math.h>

using namespace cv;
using namespace std;

const int NMAX = 100;

struct featurePoint {
	int x = 0;
	int y = 0;
	int score = 0;
	bool active = false;
};

int main();
void uint8_3_bgr_to_8_bit_gray(Mat& input_img, Mat& output_img);
void plotTracking(cv::Mat cur_img, cv::Mat prev_img, featurePoint cur_list[NMAX], featurePoint prev_list[NMAX]);
int feature_ordering(Mat image_old, Mat image_latest, featurePoint list_old[NMAX], featurePoint list_latest[NMAX], int threshold, int delta_x, int delta_y);
int feature_detection2(Mat image, featurePoint list[NMAX], double threshold);