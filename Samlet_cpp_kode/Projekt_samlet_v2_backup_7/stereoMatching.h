#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;

void getDepth(int ul, int vl, Mat gray_l, Mat gray_r,Mat disp_l , Mat disp_r, Mat M_int_left, Mat M_int_right, Mat distCoeffs1, Mat distCoeffs2,
    Mat R, Mat T, Mat R1, Mat R2, Mat P1, Mat P2, double& x, double& y, double& z, int plotStereoMatching);
void loadParams(cv::Mat& M_int_left, cv::Mat& M_int_right,
    cv::Mat& distortionCoeffs1, cv::Mat& distortionCoeffs2,
    cv::Mat& R, cv::Mat& T, cv::Mat& R1, cv::Mat& R2, cv::Mat& P1, cv::Mat& P2);