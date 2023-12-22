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

//MARTINS MOTION ESTIMAION(gradient decent):----------------------------------------------------------------
double cost_function(Mat T, Mat point_3D_homo, Mat point_3D_prev_homo) {
	//cost_function = @(T)norm(T * homogeneous_A - homogeneous_B, 'fro') ^ 2;
	return pow(norm(T * point_3D_homo - point_3D_prev_homo, cv::NORM_L2), 2);//cv::NORM_L2 flag to compute the Frobenius norm
}



Mat motionEstimationGradientDecent(Mat point_3D, Mat point_3D_prev) {
	int num_points = point_3D.cols;
	Mat point_3D_homo = point_3D;//B
	Mat point_3D_prev_homo = point_3D_prev;//A
	Mat point_3D_prev_homo_trans;//A
	Mat onesMat = Mat::ones(1, num_points, CV_64F);
	point_3D_homo.push_back(onesMat);
	point_3D_prev_homo.push_back(onesMat);
	transpose(point_3D_prev_homo, point_3D_prev_homo_trans);
	int numIter = 80000;//200
	Mat T_estimate = cv::Mat::eye(4, 4, CV_64F);// Initialize transformation matrix
	Mat error, gradient;
	double learning_rate, current_cost;
	double initial_learning_rate = 0.0002;

	//cout << "\nsum B: " <<sum(point_3D)<<" B_homo= "<< sum(point_3D_homo) <<" sum(A-B): "<< sum(point_3D_prev_homo-point_3D_homo) << "\nsum(I*A-B): " << sum(T_estimate*point_3D_prev_homo - point_3D_homo);
	for (int iter = 1;iter <= numIter;iter++) {
		error = T_estimate * point_3D_prev_homo - point_3D_homo;
		//cout << "\nsum error: " <<sum(error) ;

		//Compute gradient using the chain rule
		gradient = 2 * error * point_3D_prev_homo_trans / num_points;
		//cout << "\niter= " + to_string(iter) + " gradient: " << gradient;

		//Adaptive learning rate
		learning_rate = initial_learning_rate * log(iter);// Decrease change in rate over time
		//cout << "\niter= "+to_string(iter)+" learning_rate: " + to_string(learning_rate)+"\n";

		//Update transformation matrix with a smaller learning rate
		T_estimate = T_estimate - gradient * learning_rate;// Adjust the learning rate
		//cout << "\T_estimate: " << T_estimate;

		//calculate current cost:
		current_cost = cost_function(T_estimate, point_3D_homo, point_3D_prev_homo);

		//tjek convergence:
		if (current_cost < pow(10, -8)) {
			break;
			cout << "\nMotion estimation done after " + to_string(iter) + " iterations\n";
		}
	}

	//cout << "cost_function(\n "<<cost_function(T_estimate, point_3D_homo, point_3D_prev_homo);
	return T_estimate;

}





