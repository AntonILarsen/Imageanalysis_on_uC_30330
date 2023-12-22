// opencv_helloworld.cpp : Defines the entry point for the console application.
#include <stdio.h>
#include <iostream>
#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;
//int main()
//{
//	Mat output;
//	VideoCapture cap(0); //Try changing to 1 if not working
//	//Check if camera is available
//	if (!cap.isOpened())
//	{
//		cout << "Could not initialize capturing...\n";
//		return 0;
//	}
//	//Show Camera output
//	while (1) {
//		cap >> output;
//		imshow("webcam input", output);
//		char c = (char)waitKey(10);
//		if (c == 27) break; //Press escape to stop program
//	}
//	return 0;
//}