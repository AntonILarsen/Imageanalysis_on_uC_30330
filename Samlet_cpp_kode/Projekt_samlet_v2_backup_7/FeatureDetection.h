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

const int NMAX = 400;
// For the sake of the other guys, we must use an odd X odd kernel size. We shall return the top left coordinate of the kernel plus (kernel size - 1)/2

//tuning parameters:
const double binScale = 1;//4
const int x_bin_n = 45* binScale;//45; //Variables for setting the number f x and y bins in which
const int y_bin_n = 34* binScale;//34; //we find the RMS value to judge feasibility of feature recognition
const double stdDevRomove = 1;// remove everithing bigger than stdDevRomove*stdDev.

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
vector<cv::Mat> readImagesFromFolder(string folder_path, float factor);