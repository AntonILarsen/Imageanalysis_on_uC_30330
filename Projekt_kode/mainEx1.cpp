#include "ex1.h"
#include "motionEsti.h"
#include "FeatureDetection.h"
#include "stereoMatching.h"
#include <filesystem>

//Scales all images to x*UNI_SCALE, y*UNISCALE
//Must also be set in StereoMatching.cpp
//NB! Remember to tune SAD thresholds when changing UNI_SCALE, as SAD patches are also scaled
const float UNI_SCALE = 0.25;

//tuning variables can be found in FeatureDetection.h
namespace fs = std::filesystem;//if this doesn't work go to solution explorer->right click your project->propeties->general propetis->c++ language standat->iso c++ 17
using namespace cv;
const int LiveFeedOn = 0; //=0 for using stereo data set,1= using your 
const int plotStereoMatching = 1; 
const int useOpenCV = 0; //use openCV for the feature detection and tracking.
void plotData(const cv::Mat data, int plotWidth, int plotHeight);
void detectFeatures(const cv::Mat& image, std::vector<cv::Point2f>& points, int maxCorners, double qualityLevel, double minDistance);

void plot3DFromTop(const cv::Mat data, int plotWidth, int plotHeight);
void plot3DFromTop_2(const cv::Mat data, const cv::Mat data2, int plotWidth, int plotHeight);
void kabsch_function(Mat points_old_in, Mat points_new_in, Mat Tran, Mat Rot);
void plotAngle(Mat dst, double alpha, double beta, double gamma);


int main() {

	//Load camera intrinsincs and distortion coefficients 
	Mat M_int_left, M_int_right, distortionCoeffs1, distortionCoeffs2, R, T, R1, R2, P1, P2;
	loadParams(M_int_left, M_int_right, distortionCoeffs1, distortionCoeffs2, R, T, R1, R2, P1, P2);

	//Read images:--------------------------------------------------------------------------------
	
	string folder_path_L = "C:/Users/Willi/Downloads/Billeder/LigeLinje/left/";
	string folder_path_R = "C:/Users/Willi/Downloads/Billeder/LigeLinje/right/";
	vector<Mat> imageArray_L = readImagesFromFolder(folder_path_L, UNI_SCALE);
	vector<Mat> imageArray_R = readImagesFromFolder(folder_path_R, UNI_SCALE);

	vector<Mat> arrayCopy_L(imageArray_L);
	vector<Mat> arrayCopy_R(imageArray_R);

	double alpha = 1.5;
	double betaVal = -10 * alpha;

	for (int i = 0; i < imageArray_L.size(); i++)
	{
		cout << "Changing contrast and brightness of image " << i << " of " << imageArray_L.size() << '\n';
		arrayCopy_L[i] = imageArray_L[i].clone();
		arrayCopy_R[i] = imageArray_R[i].clone();
		
		for (int y = 0; y < arrayCopy_L[i].rows; y++) {
			for (int x = 0; x < arrayCopy_L[i].cols; x++) {
				int c = 0;
					imageArray_L[i].at<uchar>(y, x) =
						(uchar)15*sqrt(saturate_cast<uchar>(alpha * arrayCopy_L[i].at<uchar>(y, x) + betaVal));
					imageArray_R[i].at<uchar>(y, x) =
						(uchar)15*sqrt(saturate_cast<uchar>(alpha * arrayCopy_R[i].at<uchar>(y, x) + betaVal));
			}
		}
	}

	//FEATURE DETECTION AND TRACKING------------------------------------------------------------------
	Mat output;
	Mat image_1;
	Mat image_2;

	string image_path_L1 = "C:/Users/Willi/Downloads/Billeder/LigeLinje/left/left_image_00.bmp";
	Mat img_L1 = imread(image_path_L1, IMREAD_COLOR);

	cv::resize(img_L1, output, cv::Size(), UNI_SCALE, UNI_SCALE);

	Mat BW_image = Mat::zeros(Size(output.cols, output.rows), CV_8UC1);
	uint8_3_bgr_to_8_bit_gray(output, BW_image);
	image_1 = BW_image.clone();
	image_2 = BW_image.clone();

	int x_res = BW_image.cols; //Variables for storing the resolution of the image
	int y_res = BW_image.rows;

	int delta_x = (int)floor(x_res / x_bin_n); 
	int delta_y = (int)floor(y_res / y_bin_n);
	int half_delta_x = (delta_x - 1) / 2; //For odd values of delta
	int half_delta_y = (delta_y - 1) / 2;

	Mat binary_image_mat = Mat::zeros(Size(x_res, y_res), CV_8UC1);
	Mat zero_boi = binary_image_mat.clone();

	featurePoint list[NMAX], list_Old[NMAX];

	//Init parameters for stereo matching and motion estimation
	Mat H_Trans = (Mat_<double>(4, 4) <<1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);// homogenius transformation matrix
	Mat H_Trans_cum = H_Trans;
	Mat H_Rotations = (Mat_<double>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
	Mat starting_Position = (Mat_<double>(4, 1)<<0,0,0,1); //homogeneus starting pos
	Mat camera_path = (Mat_<double>(4, 1) << 0, 0, 0, 1); //homogeneus starting pos
	Mat _3D_points = Mat_<double>(3, 0); // Assuming you want a single column with 4 rows
	Mat _3D_points_prev = Mat_<double>(3, 0);
	Mat currentPoint,prevPoint;
	Mat disp_l, disp_r;
	int ul, vl,ul_prev, vl_prev; //x and y in the image plane/x til højre, y ned
	double x, y, z,x_prev,y_prev,z_prev; //outputs for the stereo matching
	vector<double> x_plot = { 0.0 }; // used for plotting the camera path
	vector<double> y_plot = { 0.0 }; // used for plotting the camera path
	vector<double> z_plot = { 0.0 }; // used for plotting the camera path

	//init for openCV tracker:
	cv::Mat old_frame, old_gray;
	std::vector<cv::Point2f> p0, p1;
	old_frame= img_L1;
	cv::cvtColor(old_frame, old_gray, cv::COLOR_BGR2GRAY);
	detectFeatures(old_gray, p0,  100,  0.01,  10);
	cv::Mat mask = cv::Mat::zeros(old_frame.size(), old_frame.type());

	const int featureRefreshInterval = 5;
	int frameCounter = 0;

	int i = 0;//image index
	while (1) {

		i++;
		if (i == imageArray_L.size()) break;

		output = imageArray_L[i];

		if (!useOpenCV) {//------------HOMEMADE FEATURE DETACTION AND TRACKING----------------
			//uint8_3_bgr_to_8_bit_gray(output, BW_image);
			GaussianBlur(output, image_1, Size(5, 5), 0); //Totally mandatory in the precense of noise

			//Finds good corner features using opencv FAST and puts the coordinates into list_Old
			feature_detection2(image_1, list, 20);

			//Finds matches of features listed in list and list_Old in image_2 and image_1, respectively,
			//and reorders list_Old, such that matching features have the same index
			feature_ordering(image_2, image_1, list_Old, list, 320, delta_x, delta_y);//200

			plotTracking(image_1, image_2, list, list_Old); //Plots lines between new and old features

			image_2 = image_1.clone();  //Set image_2 (the old image) equal to image_1, such that we have a succesion of images

			feature_detection2(image_2, list_Old, 20);
		}

		std::vector<cv::Point2f> good_old;
		std::vector<cv::Point2f> good_new;
		std::vector<uchar> inliers;
		if (useOpenCV) {//------------OpenCV FEATURE DETACTION AND TRACKING------------------
			cv::Mat frame, frame_gray;
			frame=imageArray_L[i];
			if (frame.empty()) break;
			cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

			std::vector<uchar> status;
			std::vector<float> err;
			cv::calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err);

			for (size_t j = 0; j < p0.size(); j++) {
				if (status[j]) {
					good_old.push_back(p0[j]);
					good_new.push_back(p1[j]);
				}
			}

			// RANSAC to filter out outliers
			cv::findFundamentalMat(good_old, good_new, cv::FM_RANSAC, 3, 0.99, inliers);

			// Draw the tracks for inliers
			for (size_t j = 0; j < good_new.size(); j++) {
				if (inliers[j]) {
					cv::line(mask, good_new[j], good_old[j], cv::Scalar(0, 255, 0), 2);
					cv::circle(frame, good_new[j], 5, cv::Scalar(0, 255, 0), -1);
				}
			}

			cv::add(frame, mask, frame);
			cv::imshow("Frame", frame);

			old_gray = frame_gray.clone();
			p0 = good_new;

			// New features
			detectFeatures(frame_gray, p0,  100, 0.01, 10);
			mask = cv::Mat::zeros(old_frame.size(), old_frame.type());
		}
		
		//------Stereo Matching-------------------------------------
		if (!useOpenCV) {
			//Delete previous point clouds, displayed images, and reinitialize.
			_3D_points.release();
			_3D_points_prev.release();
			disp_l.release();
			disp_r.release();
			_3D_points = Mat_<double>(3, 0);
			_3D_points_prev = Mat_<double>(3, 0);
			//Blur to help SAD in stereo matching
			GaussianBlur(imageArray_L[i], imageArray_L[i], Size(5, 5), 0);
			GaussianBlur(imageArray_R[i], imageArray_R[i], Size(5, 5), 0);
			disp_l = imageArray_L[i].clone();
			disp_r = imageArray_R[i].clone();
			//Convert images back to BGR to plot colored lines between stereo matches
			cv::cvtColor(disp_l, disp_l, COLOR_GRAY2BGR);
			cv::cvtColor(disp_r, disp_r, COLOR_GRAY2BGR);
			
			for (int j = 0; j < NMAX; j++) {
				if (list[j].active && list_Old[j].active) {
					ul = list[j].x;
					vl = list[j].y;
					ul_prev = list_Old[j].x;
					vl_prev = list_Old[j].y;
					getDepth(ul, vl, imageArray_L[i], imageArray_R[i],disp_l,disp_r, M_int_left, M_int_right, distortionCoeffs1, distortionCoeffs2, R, T, R1, R2, P1, P2, x, y, z, plotStereoMatching);
					getDepth(ul_prev, vl_prev, imageArray_L[i - 1], imageArray_R[i - 1], disp_l, disp_r, M_int_left, M_int_right, distortionCoeffs1, distortionCoeffs2, R, T, R1, R2, P1, P2, x_prev, y_prev, z_prev, !plotStereoMatching);
					currentPoint = (Mat_<double>(3, 1) << x, y, z);
					prevPoint = (Mat_<double>(3, 1) << x_prev, y_prev, z_prev);
					// Checks that points are in front of camera, no further than 10 meters in z direction away, 
					// and that z hasnt changed more than 1 meter
					if (z > 0 && z < 10 && z_prev>0 && z_prev < 10 && abs(z - z_prev) < 1) 
					{
						hconcat(_3D_points, currentPoint, _3D_points); //puts all the points into a matrix, each colum=[x;y;z]
						hconcat(_3D_points_prev, prevPoint, _3D_points_prev); //puts all the points into a matrix, each colum=[x;y;z]
					}
				}
			}
		}
		
		if (useOpenCV) {
			for (size_t j = 0; j < good_new.size(); j++) {
				if(good_new[j].x > 50 && good_new[j].y > 50 && good_old[j].x > 50 && good_old[j].y > 50)//deletes edge cases
				if (inliers[j]) {
					ul = (int)good_new[j].x;
					vl = (int)good_new[j].y;
					ul_prev = (int)good_old[j].x;
					vl_prev = (int)good_old[j].y;
					getDepth(ul, vl, imageArray_L[i], imageArray_R[i], disp_l, disp_r, M_int_left, M_int_right, distortionCoeffs1, distortionCoeffs2, R, T, R1, R2, P1, P2, x, y, z, plotStereoMatching);
					getDepth(ul_prev, vl_prev, imageArray_L[i - 1], imageArray_R[i - 1], disp_l, disp_r, M_int_left, M_int_right, distortionCoeffs1, distortionCoeffs2, R, T, R1, R2, P1, P2, x_prev, y_prev, z_prev, plotStereoMatching);
					currentPoint = (Mat_<double>(3, 1) << x, y, z);
					prevPoint = (Mat_<double>(3, 1) << x_prev, y_prev, z_prev);
					if (z > 0 && z < 50 && z_prev>0 && z_prev < 50) {

						hconcat(_3D_points, currentPoint, _3D_points); //puts all the points into a matrix, each colum=[x;y;z]
						hconcat(_3D_points_prev, prevPoint, _3D_points_prev); //puts all the points into a matrix, each colum=[x;y;z]
					}
				}
			}
		}

		Mat rot = (Mat_<double>(4, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

		if(1){
		//-----Motion Estimation------------------------------------

		if (i >= 2) {

			if (_3D_points.cols and _3D_points_prev.cols) { // Checks if there are points in both samples, else H_Trans is left unchanged.
				//H_Trans = motionEstimationIt(_3D_points, _3D_points_prev);
				H_Trans = motionEstimationGradientDecent(_3D_points, _3D_points_prev);
				H_Trans_cum = H_Trans_cum * H_Trans;
				starting_Position = (cv::Mat_<double>(4, 1) << H_Trans_cum.at<double>(0, 3), H_Trans_cum.at<double>(1, 3), H_Trans_cum.at<double>(2, 3), 1);

				//kabsch_function(_3D_points_prev, _3D_points, H_Trans, rot);

				//starting_Position = H_Rotations * H_Trans * starting_Position; //Comment out H_Rotations if Kabsch isn't used
				//H_Rotations = H_Rotations * rot; //Also for kabsch
				cout << starting_Position;
				hconcat(camera_path, starting_Position, camera_path);
			}
		}
			//plot motion of camera
			int plotwidth = 400;  // width of the plot
			int plotheight = 400; // height of the plot
			if (_3D_points.cols and _3D_points_prev.cols)
			{
				//plot3DFromTop_2(_3D_points, _3D_points_prev, 400, 400);
				plotData(camera_path, plotwidth, plotheight);
			}
		}

		// press 'w' to pause
		char c = (char)waitKey(10);
		if (c == 'w') {
			bool paused = true;
			char c;
			while (paused) {
				c = (char)waitKey(10);
				if (c == 'w')
					paused = false;
				else
					Sleep(1000);
			}
		}
		if (c == 27) break; //Press escape to stop program
	}

	destroyAllWindows();

	return 0;
}




// Function to plot data from a cv::Mat
void plotData(const cv::Mat data, int plotWidth, int plotHeight) {
	cout << "\n";
	cout << data;
	cv::Mat x = data.row(0); // 1st row
	cv::Mat z = data.row(2); // 3rd row
	cv::Mat camaraPath;
	cv::hconcat(x.reshape(1, x.total()), z.reshape(1, z.total()), camaraPath);
	cout << "\nCamera path";
	cout << camaraPath;
	camaraPath.convertTo(camaraPath, CV_32F);

	// Find min and max values for normalization
	double minX, maxX, minY, maxY;
	cv::minMaxLoc(camaraPath.col(0), &minX, &maxX);
	cv::minMaxLoc(camaraPath.col(1), &minY, &maxY);

	minX = -5;
	minY = -5;
	maxX = 5;
	maxY = 5;

	// Create an image to draw the plot
	cv::Mat plotImage = cv::Mat::zeros(plotHeight, plotWidth, CV_8UC3);

	// Draw the x and y axes
	cv::line(plotImage, cv::Point(0, plotHeight - 1), cv::Point(plotWidth, plotHeight - 1), cv::Scalar(255, 255, 255), 2);
	cv::line(plotImage, cv::Point(0, 0), cv::Point(0, plotHeight), cv::Scalar(255, 255, 255), 2);

	// Draw the data points
	for (int i = 0; i < camaraPath.rows - 1; i++) {
		// Normalize the points to fit within the plot dimensions
		int x1 = static_cast<int>((camaraPath.at<float>(i, 0) - minX) / (maxX - minX) * (plotWidth - 1));
		int y1 = plotHeight - 1 - static_cast<int>((camaraPath.at<float>(i, 1) - minY) / (maxY - minY) * (plotHeight - 1));
		int x2 = static_cast<int>((camaraPath.at<float>(i + 1, 0) - minX) / (maxX - minX) * (plotWidth - 1));
		int y2 = plotHeight - 1 - static_cast<int>((camaraPath.at<float>(i + 1, 1) - minY) / (maxY - minY) * (plotHeight - 1));

		cv::line(plotImage, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
	}

	// Show the plot
	cv::imshow("Plot", plotImage);

}


void plot3DFromTop(const cv::Mat data, int plotWidth, int plotHeight) {
	cv::Mat x = data.row(0); // 1st row
	cv::Mat y = data.row(1); // 2nd row
	cv::Mat z = data.row(2); // 3rd row

	cout << "y: " << y << '\n';

	// Find min and max values for normalization

	double minX, maxX, minY, maxY, minZ, maxZ, minG, maxG;
	cv::minMaxLoc(x, &minX, &maxX);
	cv::minMaxLoc(y, &minY, &maxY);
	cv:minMaxLoc(z, &minZ, &maxZ);

	if (minX < minZ)
		minG = minX;
	else
		minG = minZ;

	if (maxX > maxZ)
		maxG = maxX;
	else
		maxG = maxZ;

	maxG = 50;
	minG = -50;

	// Create an image to draw the plot
	cv::Mat plotImage = cv::Mat::zeros(plotHeight, plotWidth, CV_8UC3);

	// Draw the x and y axes
	cv::line(plotImage, cv::Point(0, plotHeight - 1), cv::Point(plotWidth, plotHeight - 1), cv::Scalar(255, 255, 255), 2);
	cv::line(plotImage, cv::Point(0, 0), cv::Point(0, plotHeight), cv::Scalar(255, 255, 255), 2);

	int numTicks = 5;
	for (int i = 0; i <= numTicks; i++)
	{
		cv::line(plotImage, cv::Point(0, plotHeight - i*plotHeight/(numTicks)), cv::Point(plotWidth*0.03, plotHeight - i * plotHeight / (numTicks)), cv::Scalar(255, 255, 255), 2);
	}

	// Draw the data points
	for (int i = 0; i < data.cols - 1; i++) {

		int colorBoi = (int)70 * (y.at<double>(0, i));
		// Normalize the points to fit within the plot dimensions
		int x1 = static_cast<int>((x.at<double>(0, i) - minG) / (maxG - minG) * (plotWidth - 1));
		int z1 = plotHeight - 1 - static_cast<int>((z.at<double>(0, i) - minG) / (maxG - minG) * (plotHeight - 1));
		circle(plotImage, Point(x1, z1), 3, Scalar(abs(colorBoi), colorBoi < 0 ? 0 : abs(colorBoi), colorBoi < 0 ? 0 : abs(colorBoi)), FILLED, LINE_4);

		//cv::line(plotImage, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
	}

	// Show the plot
	cv::imshow("Plot", plotImage);
}

void plot3DFromTop_2(const cv::Mat data, const cv::Mat data2, int plotWidth, int plotHeight) {
	cv::Mat x = data.row(0); // 1st row
	cv::Mat y = data.row(1); // 2nd row
	cv::Mat z = data.row(2); // 3rd row

	cv::Mat x2 = data2.row(0); // 1st row
	cv::Mat y2 = data2.row(1); // 2nd row
	cv::Mat z2 = data2.row(2); // 3rd row

	cout << "y: " << y << '\n';
	cout << "Length of data: " << size(x) << "  Length of data2: " << size(x2) << '\n';

	// Find min and max values for normalization

	double minX, maxX, minY, maxY, minZ, maxZ, minG, maxG;
	cv::minMaxLoc(x, &minX, &maxX);
	cv::minMaxLoc(y, &minY, &maxY);
	cv:minMaxLoc(z, &minZ, &maxZ);

	if (minX < minZ)
		minG = minX;
	else
		minG = minZ;

	if (maxX > maxZ)
		maxG = maxX;
	else
		maxG = maxZ;

	maxG = 5;
	minG = -5;

	// Create an image to draw the plot
	cv::Mat plotImage = cv::Mat::zeros(plotHeight, plotWidth, CV_8UC3);

	// Draw the x and y axes
	cv::line(plotImage, cv::Point(0, plotHeight - 1), cv::Point(plotWidth, plotHeight - 1), cv::Scalar(255, 255, 255), 2);
	cv::line(plotImage, cv::Point(0, 0), cv::Point(0, plotHeight), cv::Scalar(255, 255, 255), 2);

	int numTicks = 5;
	for (int i = 0; i <= numTicks; i++)
	{
		cv::line(plotImage, cv::Point(0, plotHeight - i * plotHeight / (numTicks)), cv::Point(plotWidth * 0.03, plotHeight - i * plotHeight / (numTicks)), cv::Scalar(255, 255, 255), 2);
	}

	// Draw the data points
	for (int i = 0; i < data.cols - 1; i++) {

		int colorBoi = (int)70 * (y.at<double>(0, i));

		// Normalize the points to fit within the plot dimensions
		int x1 = static_cast<int>((x.at<double>(0, i) - minG/2) / (maxG/2 - minG/2) * (plotWidth - 1));
		int z1 = plotHeight - 1 - static_cast<int>((z.at<double>(0, i) - 0) / (maxG - 0) * (plotHeight - 1));
		circle(plotImage, Point(x1, z1), 3, Scalar(255,0,0), FILLED, LINE_4);

		int x2a = static_cast<int>((x2.at<double>(0, i) - minG/2) / (maxG/2 - minG/2) * (plotWidth - 1));
		int z2a = plotHeight - 1 - static_cast<int>((z2.at<double>(0, i) - 0) / (maxG - 0) * (plotHeight - 1));
		circle(plotImage, Point(x2a, z2a), 3, Scalar(0,0,255), FILLED, LINE_4);

	}

	// Show the plot
	cv::imshow("Plot", plotImage);
}

void detectFeatures(const cv::Mat& image, std::vector<cv::Point2f>& points, int maxCorners = 100, double qualityLevel = 0.01, double minDistance = 10) {
	cv::goodFeaturesToTrack(image, points, maxCorners, qualityLevel, minDistance, cv::Mat(), 3, false, 0.04);
}

int openCVfeatureDetectionAndTracking() {
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		std::cerr << "Could not open camera" << std::endl;
		return -1;
	}

	cv::Mat old_frame, old_gray;
	std::vector<cv::Point2f> p0, p1;

	cap >> old_frame;
	cv::cvtColor(old_frame, old_gray, cv::COLOR_BGR2GRAY);
	detectFeatures(old_gray, p0);

	cv::Mat mask = cv::Mat::zeros(old_frame.size(), old_frame.type());

	const int featureRefreshInterval = 5;
	int frameCounter = 0;

	while (true) {
		cv::Mat frame, frame_gray;
		cap >> frame;
		if (frame.empty()) break;
		cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

		std::vector<uchar> status;
		std::vector<float> err;
		cv::calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err);

		std::vector<cv::Point2f> good_old;
		std::vector<cv::Point2f> good_new;
		for (size_t i = 0; i < p0.size(); i++) {
			if (status[i]) {
				good_old.push_back(p0[i]);
				good_new.push_back(p1[i]);
			}
		}

		// RANSAC to filter out outliers
		std::vector<uchar> inliers;
		cv::findFundamentalMat(good_old, good_new, cv::FM_RANSAC, 3, 0.99, inliers);

		// Draw the tracks for inliers
		for (size_t i = 0; i < good_new.size(); i++) {
			if (inliers[i]) {
				cv::line(mask, good_new[i], good_old[i], cv::Scalar(0, 255, 0), 2);
				cv::circle(frame, good_new[i], 5, cv::Scalar(0, 255, 0), -1);
			}
		}

		cv::add(frame, mask, frame);
		cv::imshow("Frame", frame);

		old_gray = frame_gray.clone();
		p0 = good_new;

		// New features
		detectFeatures(frame_gray, p0);
		mask = cv::Mat::zeros(old_frame.size(), old_frame.type());

	}

	cap.release();
	cv::destroyAllWindows();

	return 0;
}



void kabsch_function(Mat points_old_in, Mat points_new_in, Mat Tran, Mat Rot)
{
	//point_xxx_in are transposed to fit to the shape/format of points Kabsch uses internally for calculation
	//We transpose the incoming points_xxx_in
	Mat points_old = Mat::zeros(Size(points_old_in.cols, points_old_in.rows), CV_64F);
	Mat points_new = Mat::zeros(Size(points_new_in.cols, points_new_in.rows), CV_64F);
	Mat points_old_transpose = Mat::zeros(Size(points_old_in.rows, points_old_in.cols), CV_64F);
	Mat points_new_transpose = Mat::zeros(Size(points_old_in.rows, points_old_in.cols), CV_64F);

	transpose(points_old_in, points_old);
	transpose(points_new_in, points_new);

	points_old_transpose = points_old_in.clone(); //Since input is already transposed, we just copy it
	points_new_transpose = points_new_in.clone();

	//Calculating the average coordinate of all points (centroid!)
	Mat points_old_sum = (Mat_<double>(1, 3) << 0.0, 0.0, 0.0);
	Mat points_new_sum = points_old_sum.clone();

	for (int i = 0; i < points_old.rows; i++)
	{
		points_old_sum.at<double>(0, 0) += points_old.at<double>(i, 0);
		points_old_sum.at<double>(0, 1) += points_old.at<double>(i, 1);
		points_old_sum.at<double>(0, 2) += points_old.at<double>(i, 2);

		points_new_sum.at<double>(0, 0) += points_new.at<double>(i, 0);
		points_new_sum.at<double>(0, 1) += points_new.at<double>(i, 1);
		points_new_sum.at<double>(0, 2) += points_new.at<double>(i, 2);
	}
	Mat points_old_avg = points_old_sum / points_old.rows;
	Mat points_new_avg = points_new_sum / points_new.rows;

	//Finding the difference of centroid between the old and new set of points
	Mat centroid_diff = points_new_avg - points_old_avg;

	//Subtracting the respective centroids from the point clouds to centre them around 0
	for (int i = 0; i < points_old.rows; i++)
	{
		points_old.at<double>(i, 0) -= points_old_avg.at<double>(0, 0);
		points_old.at<double>(i, 1) -= points_old_avg.at<double>(0, 1);
		points_old.at<double>(i, 2) -= points_old_avg.at<double>(0, 2);

		points_new.at<double>(i, 0) -= points_new_avg.at<double>(0, 0);
		points_new.at<double>(i, 1) -= points_new_avg.at<double>(0, 1);
		points_new.at<double>(i, 2) -= points_new_avg.at<double>(0, 2);
	}

	Mat U = Mat::zeros(Size(3, 3), CV_64F);
	Mat U_transpose = U.clone();
	Mat V = U.clone();
	Mat S = U.clone();

	transpose(points_new, points_new_transpose); //Transposing the centered points_old
	
	Mat H = points_new_transpose * points_old;

	SVD::compute(H, S, U, V); //Computing the SVD to find rotation matrix

	transpose(V, V);

	transpose(U, U_transpose);
	
	double d = -1.0 + 2 * (determinant(V * U_transpose) > 0.0); //Signum function
	Mat signumMatrix = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, d);

	//cout << "U = " << endl << " " << U << endl << endl;
	//cout << "V = " << endl << " " << V << endl << endl;
	//cout << "S = " << endl << " " << S << endl << endl;
	//cout << "d = " << endl << " " << d << endl << endl;

	Mat R = V * signumMatrix * U_transpose; // Rotation

	Mat H_out_temp = (Mat_<double>(4, 4) <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), 0,
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), 0,
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), 0,
		0.0, 0.0, 0.0, 1.0);
	Mat H_inv;
;
	invert(H_out_temp, H_inv);

	transpose(points_new_avg, points_new_avg);

	Mat onesMat = Mat::ones(1, points_new_avg.cols, CV_64F);
	cv::vconcat(points_new_avg, onesMat, points_new_avg);
	transpose(points_old_avg, points_old_avg);
	cv::vconcat(points_old_avg, onesMat, points_old_avg);

	//convert points_new_avg from position vector to a homogenous translation matrix n_avg
	Mat n_avg = Mat::eye(4,4, CV_64F);
	n_avg.at<double>(0, 3) = points_new_avg.at<double>(0);
	n_avg.at<double>(1, 3) = points_new_avg.at<double>(1);
	n_avg.at<double>(2, 3) = points_new_avg.at<double>(2);
	Mat n_avg_inv = Mat::eye(4, 4, CV_64F);;
	//invert it
	invert(n_avg, n_avg_inv);

	//1. points_old_avg		-> Translate to centroid in c1 coord (world coord)
	//2. H_inv				-> Rotate to c2 coord (rotation around origo,i.e. c1 position) 
	//					       (inverse because the apparent point cloud rotation is inverse of the camera rotation)
	//3. n_avg_inv			-> Translate back from centroid to c2 in c2 coord
	//4. H_out_temp			-> Rotate back to c1/world coord.

	Mat Trans = n_avg_inv * H_inv * points_old_avg;
	
	Mat Trans2;
	transpose(Trans, Trans2);

	Mat TRANSLATION = (Mat_<double>(4, 4) <<
		1.0, 0.0, 0.0, Trans2.at<double>(0, 0),
		0.0, 1.0, 0.0, Trans2.at<double>(0, 1),
		0.0, 0.0, 1.0, Trans2.at<double>(0, 2),
		0.0, 0.0, 0.0, 1.0);

	TRANSLATION.copyTo(Tran);
	H_out_temp.copyTo(Rot);

}


//Plots three angles in dst as red, green and blue arrows, alpha, beta and gamma are in degrees
void plotAngle(Mat dst, double alpha, double beta, double gamma) {
	int rows = dst.rows;
	int cols = dst.cols;
	int x1, x2, y1, y2;
	double scale = 3.14159 / 180;
		alpha *= scale;
		beta *= scale;
		gamma *= scale;
	Scalar_<int> red = Scalar(255, 0, 0);
	Scalar_<int> green = Scalar(0, 255, 0);
	Scalar_<int> blue = Scalar(0, 0, 255);

	int length = 30;
	 
	x1 = 1 * (rows / 4);
	y1 = (rows / 2);
	x2 = x1+cos(alpha) * length;
	y2 = y1+sin(alpha) * length;
	arrowedLine(dst, Point(x1, dst.rows - 1 - y1), Point(x2, dst.rows - 1 - y2), red, 2);

	x1 = 2 * (rows / 4);
	y1 = (rows / 2);
	x2 = x1+cos(beta) * length;
	y2 = y1+sin(beta) * length;
	arrowedLine(dst, Point(x1, dst.rows - 1 - y1), Point(x2, dst.rows - 1 - y2), green, 2);

	x1 = 3 * (rows / 4);
	y1 = (rows / 2);
	x2 = x1+cos(gamma) * length;
	y2 = y1+sin(gamma) * length;
	arrowedLine(dst, Point(x1, dst.rows - 1 - y1), Point(x2, dst.rows - 1 - y2), blue, 2);
	
}