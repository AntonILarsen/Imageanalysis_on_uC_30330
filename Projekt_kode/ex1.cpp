#include "ex1.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
using namespace cv;
using namespace std;

void rectifyStereo(cv::InputArray grayL, cv::InputArray grayR, cv::OutputArray rectL, cv::OutputArray rectR);

void helloWorld() {

	cout << "hello world \n";
}

void imgInfo(string filename){

	string image_path = "C:/Users/User/OneDrive/DTU/DTU 4-2-2020/DTU/7. semester/img analysis on mc/";
	Mat img = imread(image_path+filename, IMREAD_COLOR);
	cout << "FileName: " + filename + "\nPath: " + image_path+"\nImage width x heigh: "+to_string(img.size().width)+" "+to_string(img.size().height)+"\n";
	cout << "Image depth: " + to_string(img.depth()) + "\n";
	cout << "Image type: " + to_string(img.type()) + "\n";

	while (1) {
		namedWindow("ROOCK",WINDOW_AUTOSIZE);
		moveWindow("ROOCK", 200, 200);
		resizeWindow("ROOCK", 600, 600);
		imshow("ROOCK", img);
		char c = (char)waitKey(10);
		if (c == 27) break; //Press escape to stop program
	}
	
}

void imgThres() {
	string image_path = "C:/Users/User/OneDrive/DTU/DTU 4-2-2020/DTU/7. semester/img analysis on mc/Excercises/Exercises/PEN.pgm";
	//C:\Users\User\OneDrive\DTU\DTU 4-2-2020\DTU\7. semester\img analysis on mc\Excercises\Exercises
	Mat img = imread(image_path, IMREAD_UNCHANGED);
	normalize(img, img, 0, 255, NORM_MINMAX);
	img.convertTo(img, CV_8U);
	Mat imgCopy = img;
	float width = 300; //pixel width
	double scale = float(width) / img.size().width;
	resize(img, img, cv::Size(0, 0), scale, scale);

	// Calculate the histogram of the grayscale image
	int histSize = 256;  // Number of bins in the histogram
	float range[] = { 0, 256 };  // Range of pixel values
	const float* histRange = { range };
	cv::Mat hist;
	cv::calcHist(&img, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
	// Display the histogram
	int histWidth = 512;
	int histHeight = 400;
	int binWidth = cvRound((double)histWidth / histSize);
	cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(0, 0, 0));
	// Normalize the histogram
	cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	for (int i = 1; i < histSize; i++) {
		line(histImage, cv::Point(binWidth * (i - 1), histHeight - cvRound(hist.at<float>(i - 1))),
			cv::Point(binWidth * (i), histHeight - cvRound(hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 2, 8, 0);
	}

	//thresholding
	Mat img_thres;
	int threshold_value = 100;
	threshold(img, img_thres, threshold_value, 255, THRESH_BINARY_INV);

	//Center of mass:
	int width_img = img_thres.size().width;
	int height_img = img_thres.size().height;
	int M_10 = 0;int M_01 = 0;int M_00 = 0;

	for (int i = 0; i < img_thres.rows; i++) {
		for (int j = 0; j < img_thres.cols; j++) {
			M_10 += i * img_thres.at<uchar>(i, j);
			M_01 += j * img_thres.at<uchar>(i, j);
			M_00 += img_thres.at<uchar>(i, j);
		}
	}

	cout << "\nM_00 :" + to_string(M_00) + "\n\n";
	cout << "\nM_01 :" + to_string(M_01) + "\n\n";
	cout << "\nM_10 :" + to_string(M_10) + "\n\n";
	int y = M_10 / M_00;
	int x = M_01 / M_00;
	cout << "\nx= y= " + to_string(x) + " " + to_string(y) + "\n\n";
	int crosshairLength = 20;
	Mat color_img;
	cvtColor(img_thres, color_img, COLOR_GRAY2RGB);
	// Draw horizontal line (crosshair)
	cv::line(color_img, cv::Point(x - crosshairLength, y), cv::Point(x + crosshairLength, y), cv::Scalar(255,0, 0), 2);

	// Draw vertical line (crosshair)
	cv::line(color_img, cv::Point(x, y - crosshairLength), cv::Point(x, y + crosshairLength), cv::Scalar(255,0, 0), 2);

	// Display the image with the crosshair

	while (1) {
		imshow("Histogram", histImage);
		moveWindow("Histogram", 600, 0);
		imshow("PEN.pgm", img);
		moveWindow("PEN.pgm", 0, 0);
		imshow("Pen_thres", color_img);
		moveWindow("Pen_thres", 300, 0);

		char c = (char)waitKey(10);
		if (c == 27) break; //Press escape to stop program
	}
}

void imgFilter(string filename) {
	string image_path = "C:/Users/User/OneDrive/DTU/DTU 4-2-2020/DTU/7. semester/img analysis on mc/";
	Mat img = imread(image_path + filename, IMREAD_GRAYSCALE);
	//cout << "FileName: " + filename + "\nPath: " + image_path + "\nImage width x heigh: " + to_string(img.size().width) + " " + to_string(img.size().height) + "\n";
	//cout << "Image depth: " + to_string(img.depth()) + "\n";
	//cout << "Image type: " + to_string(img.type()) + "\n";
	cout << "\n\nimage rows: " + to_string(img.rows)+"\n\n";
	DWORD currentTime = GetTickCount();
	cout << "\n\nCurrent time: " + to_string(currentTime) + "\n\n";
	 
	Mat blankImage(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
	Mat img_filtered = blankImage;
	int n = 1;

	for (int i = 0+n; i < img.rows-n; i++) {
		for (int j = 0+n; j < img.cols-n; j++) {

			for (int k = i-n; k <= i+n; k++) {
				for (int l = j-n; l <= j+n; l++) {
					img_filtered.at<uint8_t>(i, j) += static_cast<uint8_t>((1.0 / pow(2 * n + 1, 2)) * static_cast<uint8_t>(img.at<uint8_t>(k, j)));

				}
			}
		}	
	}
	DWORD currentTime2 = GetTickCount();
	cout << "\n\nCurrent time: " + to_string(currentTime2) + "\n\n";
	cout << "time elapsed: " + to_string(currentTime2 - currentTime) + "ms\n\n";

	Mat kernel = (cv::Mat_<float>(3, 3) <<
		-1, -1, -1,
		-1, 8, -1,
		-1, -1, -1);

	// Create a destination image for the result
	Mat result;

	// Apply the convolution using filter2D
	cv::filter2D(img, result, -1, kernel);

	DWORD currentTime3 = GetTickCount();
	cout << "time elapsed2: " + to_string(currentTime3 - currentTime2) + "ms\n\n";

	while (1) {
		//namedWindow("ROOCK", WINDOW_AUTOSIZE);
		//moveWindow("ROOCK", 200, 200);
		//resizeWindow("ROOCK", 600, 600);
		imshow("original", img);
		imshow("Filtered", img_filtered);
		char c = (char)waitKey(10);
		if (c == 27) break; //Press escape to stop program
	}

}


void contourSearch(Mat pic, Point pos, int rimx[], int rimy[], int local_tresh) {
	int count = 0;
	Point newpos;
	int randx[RAND_MAX], randy[RAND_MAX];
	int draw_type = 0;
	newpos = pos;
	while (newpos.x >= 0 && newpos.y >= 0 && newpos.x < pic.cols && newpos.y <
		pic.rows)
	{
		// save current position in list
		rimx[count] = newpos.x;
		rimy[count] = newpos.y;
		count++;
		// Select next search direction
		draw_type = (draw_type + 6) % 8;
		switch (draw_type)
		{
		case 0:
			if (pic.at<uchar>(newpos.y, newpos.x + 1) > local_tresh) {
				newpos.x += 1;
				draw_type = 0;
				break;
			}
		case 1:
			if (pic.at<uchar>(newpos.y + 1, newpos.x + 1) > local_tresh) {
				newpos.x += 1;
				newpos.y += 1;
				draw_type = 1;
				break;
			}
		case 2:
			if (pic.at<uchar>(newpos.y + 1, newpos.x) > local_tresh) {
				newpos.y += 1;
				draw_type = 2;
				break;
			}
		case 3: if (pic.at<uchar>(newpos.y + 1, newpos.x - 1) > local_tresh) {
			newpos.x -= 1;
			newpos.y += 1;
			draw_type = 3;
			break;
		}
		case 4:
			if (pic.at<uchar>(newpos.y, newpos.x - 1) > local_tresh) {
				newpos.x -= 1;
				draw_type = 4;
				break;
			}
		case 5:
			if (pic.at<uchar>(newpos.y - 1, newpos.x - 1) > local_tresh) {
				newpos.x -= 1;
				newpos.y -= 1;
				draw_type = 5;
				break;
			}
		case 6:
			if (pic.at<uchar>(newpos.y - 1, newpos.x) > local_tresh) {
				newpos.y -= 1;
				draw_type = 6;
				break;
			}
		case 7:
			if (pic.at<uchar>(newpos.y - 1, newpos.x + 1) > local_tresh) {
				newpos.x += 1;
				newpos.y -= 1;
				draw_type = 7;
				break;
			}
		case 8:
			if (pic.at<uchar>(newpos.y, newpos.x + 1) > local_tresh) {
				newpos.x += 1;
				draw_type = 0;
				break;
			}
		case 9:
			if (pic.at<uchar>(newpos.y + 1, newpos.x + 1) > local_tresh) {
				newpos.x += 1;
				newpos.y += 1;
				draw_type = 1;
				break;
			}
		case 10:
			if (pic.at<uchar>(newpos.y + 1, newpos.x) > local_tresh) {
				newpos.y += 1;
				draw_type = 2;
				break;
			}
		case 11:
			if (pic.at<uchar>(newpos.y + 1, newpos.x - 1) > local_tresh) {
				newpos.x -= 1;
				newpos.y += 1;
				draw_type = 3;
				break;
			}
		case 12:
			if (pic.at<uchar>(newpos.y, newpos.x - 1) > local_tresh) {
				newpos.x -= 1;
				draw_type = 4;
				break;
			}
		case 13:
			if (pic.at<uchar>(newpos.y - 1, newpos.x - 1) > local_tresh) {
				newpos.x -= 1;
				newpos.y -= 1;
				draw_type = 5;
				break;
			}
		case 14:
			if (pic.at<uchar>(newpos.y - 1, newpos.x) > local_tresh) {
				newpos.y -= 1;
				draw_type = 6;
				break;
			}
		}
		// If we are back at the beginning, we declare success
		if (newpos.x == pos.x && newpos.y == pos.y)
			break;
		// Abort if the contour is too complex.
		if (count >= RAND_MAX)
			break;
	}
}

#include <opencv2/opencv.hpp>
using namespace cv;

void contourSearch2(Mat pic, Point pos, int rimx[], int rimy[], int local_tresh) {
	int count = 0;
	Point newpos = pos;
	int draw_type = 0;
	int maxCount = pic.rows * pic.cols; // Maximum allowed contour points

	while (count < maxCount) {
		rimx[count] = newpos.x;
		rimy[count] = newpos.y;
		count++;

		// Store the previous position for comparison
		Point prevpos = newpos;

		// Select next search direction
		draw_type = (draw_type + 1) % 8;

		// Calculate coordinates for the next position based on draw_type
		switch (draw_type) {
		case 0:
			newpos.x += 1;
			break;
		case 1:
			newpos.x += 1;
			newpos.y += 1;
			break;
		case 2:
			newpos.y += 1;
			break;
		case 3:
			newpos.x -= 1;
			newpos.y += 1;
			break;
		case 4:
			newpos.x -= 1;
			break;
		case 5:
			newpos.x -= 1;
			newpos.y -= 1;
			break;
		case 6:
			newpos.y -= 1;
			break;
		case 7:
			newpos.x += 1;
			newpos.y -= 1;
			break;
		}

		// Check if the new position is within image bounds
		if (newpos.x < 0 || newpos.y < 0 || newpos.x >= pic.cols || newpos.y >= pic.rows) {
			break; // Out of bounds, terminate contour tracing
		}

		// Check if the pixel value at the new position is greater than the threshold
		if (pic.at<uchar>(newpos.y, newpos.x) <= local_tresh) {
			newpos = prevpos; // Revert to the previous position
		}

		// Check if we have returned to the starting position
		if (newpos == pos) {
			break; // Contour tracing is complete
		}
	}
}

void correspondence() {
	//string image_path_R = "C:/Users/User/OneDrive/DTU/DTU 4-2-2020/DTU/7. semester/img analysis on mc//Excercises/Exercises/PIC1_R.png";
	//string image_path_L = "C:/Users/User/OneDrive/DTU/DTU 4-2-2020/DTU/7. semester/img analysis on mc//Excercises/Exercises/PIC1_L.png";
	//string image_path_L = "C:/Users/User/OneDrive/DTU/DTU 4-2-2020/DTU/7. semester/img analysis on mc//Excercises/Exercises/motorcykel_R.png";
	//string image_path_R = "C:/Users/User/OneDrive/DTU/DTU 4-2-2020/DTU/7. semester/img analysis on mc//Excercises/Exercises/motorcykel_L.png";
	//string image_path_L = "C:/Users/User/OneDrive/DTU/DTU 4-2-2020/DTU/7. semester/img analysis on mc//Excercises/Exercises/PIC1_R.png";
	
	string image_path_L = "C:/Users/User/Documents/GitHub/Imageanalysis_on_uC_30330/stereo_calib/calib_l/7l.png";
	string image_path_R = "C:/Users/User/Documents/GitHub/Imageanalysis_on_uC_30330/stereo_calib/calib_r/7r.png";
	//string image_path_L = "C:/Users/User/Documents/GitHub/Imageanalysis_on_uC_30330/stereo_calib_v2/Left/l7.png";//new calib img, men mangler nye calib værdier værdier
	//string image_path_R = "C:/Users/User/Documents/GitHub/Imageanalysis_on_uC_30330/stereo_calib_v2/Right/r7.png";
	Mat img_R = imread(image_path_R, IMREAD_GRAYSCALE);
	Mat img_L = imread(image_path_L, IMREAD_GRAYSCALE);
	Mat rect_L, rect_R;
	rectifyStereo(img_L, img_R, rect_L, rect_R);
	//resize down
	int down_width = 640; double scale = float(down_width) / img_R.size().width;
	resize(rect_R, rect_R, cv::Size(0, 0), scale, scale);
	resize(rect_L, rect_L, cv::Size(0, 0), scale, scale);

	int start_idx_x = 50;
	int start_idx_y = 300;
	int search_band_y = 10;
	int search_area = 20; //10x10
	int sum1 = 0;
	int sumBest = INT_MAX;
	int bestXY[2] = {0,0};
	int y2 = start_idx_y; // vi behøver kun at søge i x retningen, da billedet kun er forskudt i den retning.
	for (int y2 = start_idx_y- search_band_y/2; y2 < start_idx_y+ search_band_y/2; y2++) {//search img
		for (int x2 = 0; x2 < rect_L.cols-search_area; x2++) {//search img

			for (int y = 0; y < search_area; y++) {//original
				for (int x = 0; x < search_area; x++) {//original
					//sum1 += abs(rect_R.at<uint8_t>( y + start_idx_y, x + start_idx_x) - rect_L.at<uint8_t>(y + y2, x + x2));
					sum1 += pow(rect_R.at<uint8_t>(y + start_idx_y, x + start_idx_x) - rect_L.at<uint8_t>(y + y2, x + x2),2);
				}
			}
		if (sum1 < sumBest) {
			sumBest = sum1;
			bestXY[0] = x2;bestXY[1] = y2;
		}
		//cout << "\nCurrentSum: " + to_string(sum1) + " Best sum: " + to_string(sumBest) + " best x: " + to_string(bestXY[0]);
		sum1 = 0;


		}
	}
	cout << "\n\n(x , y)=  (" + to_string(bestXY[0]) + ", " + to_string(bestXY[1]) + ")\n\n";

	//depth estimation:
	double baseLine = 193.001; double f = 3997.684;double doffs = 131.111;//camera propeties
	double cx0 = 1176.728;double cy0 = 1011.728;//vi bruger kun dem fra venstre camera
	double cx1 = 1307.839;double cy1 = 1011.728;
	int disperity = (start_idx_x - bestXY[0]);
	double Z_depth_mm = baseLine * f / (disperity + doffs);
	double X_depth_mm = baseLine * (start_idx_x - cx0) / (disperity + doffs);
	double Y_depth_mm = baseLine * f*(bestXY[1]- cy0) / (f*(disperity + doffs));
	cout << "\n\n depth in meter: " + to_string(Z_depth_mm / 1000)+" Disperity: "+to_string(disperity)+" X Y= "+to_string(X_depth_mm/1000)+" "+ to_string(Y_depth_mm/1000) + "\n\n";

	int crosshairLength = search_area;
	Mat color_img_L, color_img_R;
	cvtColor(rect_L, color_img_L, COLOR_GRAY2RGB);
	cvtColor(rect_R, color_img_R, COLOR_GRAY2RGB);
	cv::line(color_img_L, cv::Point(bestXY[0] - crosshairLength, bestXY[1]), cv::Point(bestXY[0] + crosshairLength, bestXY[1]), cv::Scalar(255, 0, 0), 1);
	cv::line(color_img_L, cv::Point(bestXY[0], bestXY[1] - crosshairLength), cv::Point(bestXY[0], bestXY[1] + crosshairLength), cv::Scalar(255, 0, 0), 1);
	cv::line(color_img_R, cv::Point(start_idx_x - crosshairLength, start_idx_y), cv::Point(start_idx_x + crosshairLength, start_idx_y), cv::Scalar(255, 0, 0), 1);
	cv::line(color_img_R, cv::Point(start_idx_x, start_idx_y - crosshairLength), cv::Point(start_idx_x, start_idx_y + crosshairLength), cv::Scalar(255, 0, 0), 1);
	cv::line(color_img_L, cv::Point(0, 240), cv::Point(630, 240), cv::Scalar(255, 0, 0), 1);
	cv::line(color_img_R, cv::Point(0, 240), cv::Point(630, 240), cv::Scalar(255, 0, 0), 1);
	while (1) {
		imshow("PEN_R", color_img_R);
		moveWindow("PEN_R", 0, 0);
		imshow("PEN_L", color_img_L);
		moveWindow("PEN_L", 500, 0);
		char c = (char)waitKey(10);
		if (c == 27) break; //Press escape to stop program
	}
}


void rectifyStereo(cv::InputArray grayL, cv::InputArray grayR, cv::OutputArray rectL, cv::OutputArray rectR) {
	// Camera intrinsic parameters (adjust these based on your camera setup)
	//Mat K1, K2;       // Camera calibration matrices -> camera matrix = [fx 0 0]^T , [0 fy 0]^T , [cx cy 1]^T]

	// Camera 1 Intrinsics
	cv::Mat cameraMatrix1 = (cv::Mat_<double>(3, 3) <<
		1089.3846, 0, 627.9585,
		0, 1089.4163, 452.8457,
		0, 0, 1
		);
	cv::Mat distCoeffs1 = (cv::Mat_<double>(5, 1) << -0.3982, 0.1976, 0, 0, 0);

	// Camera 2 Intrinsics
	cv::Mat cameraMatrix2 = (cv::Mat_<double>(3, 3) <<
		1068.0402, 0, 630.6716,
		0, 1068.1012, 471.5385,
		0, 0, 1
		);
	cv::Mat distCoeffs2 = (cv::Mat_<double>(5, 1) << -0.3870, 0.1673, 0, 0, 0);

	// Camera 1 to Camera 2 Transformation
	cv::Mat R = (cv::Mat_<double>(3, 3) <<
		0.9999, -0.0098, 0.0094,
		0.0099, 0.9999, -0.0055,
		-0.0094, 0.0056, 0.9999
		);
	cv::Mat T = (cv::Mat_<double>(3, 1) << -249.3457, -0.3537, 1.3669);

	// Compute rectification matrices for the stereo pair
	cv::Mat R1, R2, P1, P2, Q;
	cv::stereoRectify(
		cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
		grayL.size(), R, T, R1, R2, P1, P2, Q);

	// Compute the rectification maps
	cv::Mat map1x, map1y, map2x, map2y;
	cv::initUndistortRectifyMap(
		cameraMatrix1, distCoeffs1, R1, P1, grayL.size(), CV_32FC1, map1x, map1y);
	cv::initUndistortRectifyMap(
		cameraMatrix2, distCoeffs2, R2, P2, grayL.size(), CV_32FC1, map2x, map2y);

	// Apply rectification to the images
	cv::remap(grayL, rectL, map1x, map1y, cv::INTER_LINEAR);
	cv::remap(grayR, rectR, map2x, map2y, cv::INTER_LINEAR);
}