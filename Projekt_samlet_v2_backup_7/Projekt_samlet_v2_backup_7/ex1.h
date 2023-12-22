#pragma once
#include <iostream>
#include<opencv2/opencv.hpp>
#include <windows.h>

using namespace std;
using namespace cv;

void helloWorld();

void imgInfo(string filename);

void imgThres();

void imgFilter(string filename);

void contourSearch(Mat pic, Point pos, int rimx[], int rimy[], int local_tresh);

void contourSearch2(Mat pic, Point pos, int rimx[], int rimy[], int local_tresh);

void correspondence();
