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


const int NMAX = 1000;

struct featurePoint {
	int x = 0;
	int y = 0;
	int score = 0;
	bool active = false;
};

int main();
int part2function(string filePathBoi);
void uint8_3_bgr_to_8_bit_gray(Mat& input_img, Mat& output_img);
void uint8_3_bgr_to_8_bit_gray_2(Mat& input_img, Mat& output_img, int channel_no);
void histogram_boi(Mat& input_img, char channel_no);
void histogram_boi_anim(Mat& input_img, char channel_no, string window_name, bool normalise = 0);
void binary_image(Mat& input_img, Mat& output_img, int threshold_val);
void center_binary(Mat &input_img, int output_coords[2]);
float pq_central_image_moment(Mat& input_img, int center_x, int center_y, int p, int q);
float eta_moment(float mu_ij, float mu_00, int i, int j);
double image_sum(Mat input_image);
double image_sum_subarea(Mat input_image, int start_x, int start_y, int delta_x, int delta_y);
void image_subset_copy(Mat input_image, Mat output_image, int start_x, int start_y, int delta_x, int delta_y);
double RMS_value(Mat input_image, bool mean_subtract);
double HS_window_classifier(Mat input_image, double k);
double HS_window_classifier2(Mat input_image, float k);
void matrix_square_root(const cv::Mat& A, cv::Mat& sqrtA); 
int feature_detection(Mat latest_image, featurePoint list[NMAX], double threshold);
