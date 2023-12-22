#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>
//#include "motionEsti.h"
using namespace cv;
using namespace std;

double SAD(Mat point_3D, Mat point_3D_prev) {
	//assumes 3 rows for x y z.
	int row = 3;
	int colm = point_3D.cols;
	double error = 0;
	for (int i = 0; i < colm; i++)
		for (int j = 0;j < row;j++)
			error = error + abs(point_3D.at<double>(j,i)- point_3D_prev.at<double>(j, i));

	
	return error;
}

Mat  H_trans(double a, double b, double c, double alpha, double beta, double gamma){
	Mat H = (Mat_<double>(4, 4) << cos(beta) * cos(gamma), cos(gamma) * sin(alpha) * sin(beta) - cos(alpha) * sin(gamma), sin(alpha) * sin(gamma) + cos(alpha) * cos(gamma) * sin(beta), a,
		cos(beta) * sin(gamma), cos(alpha) * cos(gamma) + sin(alpha) * sin(beta) * sin(gamma), cos(alpha) * sin(beta) * sin(gamma) - cos(gamma) * sin(alpha), b,
		-sin(beta), cos(beta) * sin(alpha), cos(alpha) * cos(beta), c,
		0, 0, 0, 1);
	return H;
}

void findBestVal(Mat point_3D, Mat point_3D_prev,double ink,int numIter,int optiFor,Mat startH, double *bestReturn, double *errorReturn) {
	int sel[6] = { 0,0,0,0,0,0 };
	sel[optiFor] = 1;
	int numPoints = point_3D.cols;
	int best = 0;
	double error = 1000000000;

	//for (int i = 0; i < numIter; i++) {

		Mat transform1 = H_trans(	(best + ink) * sel[0] + startH.at<double>(0, 0),
									(best + ink) * sel[1] + startH.at<double>(0, 1),
									(best + ink) * sel[2] + startH.at<double>(0, 2),
									(best + ink) * sel[3] + startH.at<double>(0, 3),
									(best + ink) * sel[4] + startH.at<double>(0, 4),
									(best + ink) * sel[5] + startH.at<double>(0, 5));
		Mat transform2 = H_trans(	(best - ink) * sel[0] + startH.at<double>(0, 0),
									(best - ink) * sel[1] + startH.at<double>(0, 1),
									(best - ink) * sel[2] + startH.at<double>(0, 2),
									(best - ink) * sel[3] + startH.at<double>(0, 3),
									(best - ink) * sel[4] + startH.at<double>(0, 4),
									(best - ink) * sel[5] + startH.at<double>(0, 5));
		Mat onesMat = Mat::ones(1, numPoints, CV_64F);
		Mat p1 = point_3D;
		p1.push_back(onesMat);

		Mat tripoint1 = transform1 * p1;
		Mat tripoint2 = transform2 * p1;

		cv::Range rowsToExtract(0, 3);  // Extract rows from index 0 to 2, (the first 3 rows) 
		double erUp = SAD(point_3D_prev,tripoint1.rowRange(rowsToExtract));
		double erDown = SAD(point_3D_prev, tripoint2.rowRange(rowsToExtract));

		if (erUp < erDown) {
			*bestReturn = best + ink+ startH.at<double>(0, optiFor);
			*errorReturn = erUp;
		}
		if (erUp > erDown) {
			*bestReturn = best - ink + startH.at<double>(0, optiFor);
			*errorReturn = erDown;
		}



	//}




}


Mat motionEstimationIt(Mat point_3D, Mat point_3D_prev) {
	
	Mat prevBest = (Mat_<double>(1,6) << 0,0,0,0,0,0 );

	double best[6] = {0,0,0,0,0,0};
	double error[6] = {0,0,0,0,0,0};
	double ink = 0.1;
	const int numIt = 100;
	double error_cum[numIt];
	for (int k = 0; k < numIt; k++) {
		for (int i = 0; i < 6; i++) {
			findBestVal(point_3D, point_3D_prev, ink, 1, i, prevBest, &best[i], &error[i]);
			 //cout << "best: " + to_string(best[i]) + " error: " + to_string(error[i]) + "\n";
			 //if (k == 12)
		}
		double minVal = error[0];
		int minIndex = 0;
		for (int i = 0; i < 6; i++)
			if (error[i] < minVal) {
				minVal = error[i];
				minIndex = i;
			}
		error_cum[k] = error[minIndex];

		//cout << "minidx "+to_string(minIndex) + "\n";
		if (k >= 2)
			if (error_cum[k] - error_cum[k - 2] == 0) {
				ink = ink / 2;
				//cout << to_string(k) + "\n";
			}
		prevBest.at<double>(0, minIndex) = best[minIndex];
		//cout << "k= " + to_string(k) + " prevBest= ";
	}
	//cout << (prevBest);
	//cout << "\n";
	return H_trans(prevBest.at<double>(0, 0) , 
		prevBest.at<double>(0, 1), 
		prevBest.at<double>(0, 2), 
		prevBest.at<double>(0, 3), 
		prevBest.at<double>(0, 4), 
		prevBest.at<double>(0, 5));
}


//MARTINS MOTION ESTIMAION:----------------------------------------------------------------
double cost_function(Mat T, Mat point_3D_homo, Mat point_3D_prev_homo) {
	//cost_function = @(T)norm(T * homogeneous_A - homogeneous_B, 'fro') ^ 2;
	return pow(norm(T * point_3D_homo - point_3D_prev_homo, cv::NORM_L2),2);//cv::NORM_L2 flag to compute the Frobenius norm
}



Mat motionEstimationGradientDecent(Mat point_3D, Mat point_3D_prev) {
	Mat onesMat = Mat::ones(1, 100, CV_64F);
	Mat point_3D_homo = point_3D;
	Mat point_3D_prev_homo = point_3D_prev;
	point_3D_homo.push_back(onesMat);
	point_3D_prev_homo.push_back(onesMat);

	// Initialize transformation matrix
	Mat T_estimate =(Mat_<double>(4, 4) <<
		1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1
	);
	cout << "cost_function(\n";
	cout << cost_function(T_estimate, point_3D_homo, point_3D_prev_homo);
	cout << "\n\n";
	return T_estimate;
}




