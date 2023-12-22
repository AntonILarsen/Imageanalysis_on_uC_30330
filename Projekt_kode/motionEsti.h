#pragma once
#include <iostream>
#include<opencv2/opencv.hpp>
#include <windows.h>

Mat motionEstimationIt(Mat point_3D, Mat point_3D_prev);

Mat motionEstimationGradientDecent(Mat point_3D, Mat point_3D_prev);//NOT DONE
