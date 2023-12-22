#include "ex1.h"
#include "motionEsti.h"
#include "FeatureDetection.h"
#include "stereoMatching.h"
#include <filesystem>

const float UNI_SCALE = 0.25;

//tuning varabels can be found in FeatureDetecting.h
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

	Mat M_int_left, M_int_right, distortionCoeffs1, distortionCoeffs2, R, T, R1, R2, P1, P2;
	loadParams(M_int_left, M_int_right, distortionCoeffs1, distortionCoeffs2, R, T, R1, R2, P1, P2);

	/*
	Mat C = (Mat_<double>(3,6) << 0.578, -1, 0, -1, 5, -1, 0, -1, 0, 0.578, -1, 0, -1, 5, -1, 0, -1, 0);
	Mat C2 = (Mat_<double>(3, 6) << 0.578, -1, 0, -1, 5, -1, 0, -1, 0, 0.578, -1, 0, -1, 5, -1, 0, -1, 1);
	Mat C3 = (Mat_<double>(3, 6) << 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18);

	Mat f1 = (Mat_<double>(3, 100) <<
		-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3.5474, 2.8049, 3.0839, 2.9044, 3.4698, 2.6803, 3.649, 2.9362, 4.1481, 3.5177, 3.4817, 3.6529, 2.728, 3.2283, 2.7059, 3.2329, 2.7611, 2.4928, 3.4686, 3.7898, 3.0324, 3.6081, 3.0387, 2.9337, 3.2174, 2.8134, 3.2484, 3.1551, 2.6046, 3.9169, 3.3645, 3.3284, 2.8598, 2.5465, 3.1847, 3.6305, 2.5345, 3.304, 2.7108, 2.7453, 2.9001, 3.1673, 3.0158, 3.4258, 2.6905, 3.6513, 2.7494, 2.756, 3.411, 3.3848, 3.2965, 2.5365, 2.7203, 3.1632, 3.009, 2.9337, 2.7342, 2.56, 3.2415, 3.0273, 3.2176, 2.6902, 3.7086, 3.4851, 3.4809, 2.9073, 3.3728, 2.9136, 3.2532, 2.7698, 3.3523, 2.4363, 2.6535, 2.9543, 2.9732, 2.2712, 3.4807, 3.8096, 3.0135, 3.0312, 3.0501, 3.4479, 3.7022, 2.6974, 3.1956, 3.3514, 3.3026, 2.6561, 3.0151, 2.3732, 2.8947, 2.1061, 2.4983, 3.3228, 3.0295, 3.1399, 2.5771, 3.5663, 2.8448, 2.6352
	);
	Mat f2 = (Mat_<double>(3, 100) <<
		-4.073, -3.3748, -2.3444, -1.4631, -0.3395, 0.3434, 1.5982, 2.306, 3.6399, 4.3745, -3.9887, -2.9933, -2.3545, -1.252, -0.4823, 0.6288, 1.415, 2.2675, 3.5246, 4.5688, -4.0291, -2.9022, -2.1477, -1.2422, -0.2102, 0.5981, 1.6793, 2.5886, 3.3492, 4.7158, -3.8154, -2.8875, -2.1002, -1.2624, -0.1151, 0.9696, 1.5527, 2.7427, 3.4895, 4.4404, -3.8608, -2.8342, -1.9438, -0.8707, -0.1702, 1.082, 1.7283, 2.6702, 3.8229, 4.7541, -3.6262, -2.9337, -1.9342, -0.8505, 0.0391, 0.9543, 1.8291, 2.7121, 3.8735, 4.7435, -3.5461, -2.778, -1.5071, -0.6401, 0.2983, 1.0514, 2.1425, 2.9328, 3.983, 4.7654, -3.3966, -2.7549, -1.7446, -0.707, 0.2388, 0.9502, 2.2833, 3.3299, 4.0107, 4.9561, -3.3893, -2.3202, -1.2977, -0.6849, 0.4168, 1.4072, 2.331, 3.0604, 4.1169, 4.8478, -3.3341, -2.6509, -1.5837, -0.3758, 0.4685, 1.4441, 2.2007, 3.4622, 4.1672, 5.0387, -5.8515, -5.622, -5.7083, -5.6528, -5.8275, -5.5835, -5.8829, -5.6626, -6.0371, -5.8423, -4.8801, -4.933, -4.6472, -4.8018, -4.6404, -4.8032, -4.6574, -4.5745, -4.8761, -4.9753, -3.7902, -3.9681, -3.7922, -3.7597, -3.8474, -3.7226, -3.857, -3.8281, -3.658, -4.0636, -2.9418, -2.9306, -2.7858, -2.689, -2.8863, -3.024, -2.6853, -2.9231, -2.7398, -2.7505, -1.8472, -1.9298, -1.883, -2.0097, -1.7825, -2.0794, -1.8007, -1.8027, -2.0051, -1.997, -1.0187, -0.7838, -0.8406, -0.9775, -0.9298, -0.9066, -0.8449, -0.7911, -1.0017, -0.9355, -0.0432, 0.1197, -0.195, -0.1259, -0.1246, 0.0526, -0.0912, 0.0507, -0.0542, 0.0951, 0.8662, 1.1493, 1.0821, 0.9892, 0.9833, 1.2003, 0.8265, 0.7249, 0.9709, 0.9654, 1.9106, 1.7877, 1.7091, 2.0196, 1.8657, 1.8175, 1.8326, 2.0324, 1.9214, 2.1198, 2.9097, 3.1534, 3.0322, 2.7774, 2.8681, 2.834, 3.0079, 2.7022, 2.9251, 2.9899, 3.7285, 2.7229, 2.6303, 2.1278, 2.2911, 1.2435, 1.7672, 0.7882, 1.5292, 0.6238, 3.9602, 3.7711, 2.6026, 2.7076, 1.8988, 2.0277, 1.264, 0.6823, 1.2123, 1.1573, 3.849, 4.0215, 3.1706, 2.7348, 2.6463, 1.9432, 1.9899, 1.5645, 0.7305, 1.5613, 4.4362, 4.0619, 3.3011, 2.6791, 2.9075, 2.9638, 1.6423, 1.988, 1.1158, 0.8047, 4.3116, 4.2083, 3.7309, 3.7553, 2.7561, 3.2728, 2.1247, 1.7886, 2.032, 1.6665, 4.9562, 3.935, 3.7572, 3.811, 3.3312, 2.9218, 2.4015, 1.9038, 2.1709, 1.6374, 5.1761, 4.3627, 4.9308, 4.3891, 4.0433, 3.1887, 3.2626, 2.5103, 2.4717, 1.6977, 5.5868, 4.4262, 4.2783, 4.2051, 3.88, 2.9105, 3.6495, 3.6013, 2.5479, 2.2217, 5.6071, 5.6206, 5.5059, 4.2659, 4.3691, 4.1663, 3.7806, 2.8608, 2.8397, 1.924, 5.7586, 4.7118, 4.7203, 5.1151, 4.511, 4.2676, 3.4227, 3.9647, 2.9779, 2.4485
		);
	Mat onesMat = Mat::ones(1,100, CV_64F);
	Mat p1 = f1;
	p1.push_back(onesMat);
	cv::Range rowsToExtract(0, 4);

	//cout << "C3 = " << endl << " " << p1.rowRange(rowsToExtract) << endl << endl;
	Mat GT = (Mat_<double>(4,4) <<
		0.9397, 0, -0.3420, 0, 0.1057, 0.9511, 0.2904, 0, 0.3253, -0.3090, 0.8937, 0, 0, 0, 0.3000, 1
		);

	
	cout << "Ground trouth:\n";
	cout << GT;
	cout << "\nAntons motion estimation:\n";
	cout<< motionEstimationIt(f1,f2);//outputs transformation matrix
	cout << "\nMartins motion estimation:\n";
	cout << motionEstimationGradientDecent(f1, f2);
	cout << "\n\n";
	*/
	//Read images:--------------------------------------------------------------------------------
	
	string folder_path_L = "C:/Users/Willi/Downloads/Billeder/stereo_video_v4/left/";
	string folder_path_R = "C:/Users/Willi/Downloads/Billeder/stereo_video_v4/right/";
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
	//VideoCapture cap(0); //Try changing to 1 if not working
	//Check if camera is available
	//if (!cap.isOpened())
	//{
	//	cout << "Could not initialize capturing...\n";
	//	return 0;
	//}

	//cap.set(CAP_PROP_EXPOSURE, -6.0);//-6
	//cap.set(CAP_PROP_FRAME_WIDTH, 320);//320/1280
	//cap.set(CAP_PROP_FRAME_HEIGHT, 240);//240/960

	//cout << "Exposure value: " << to_string(cap.get(CAP_PROP_EXPOSURE)) << '\n';
	//cout << "Gain value: " << to_string(cap.get(CAP_PROP_GAIN)) << '\n';

	//ANTON
	string image_path_L1 = "C:/Users/Willi/Downloads/Billeder/stereo_video_v4/left/l13.png";
	Mat img_L1 = imread(image_path_L1, IMREAD_COLOR);
	//output = img_L1;
	cv::resize(img_L1, output, cv::Size(), UNI_SCALE, UNI_SCALE);
	//ANTON
	//if(LiveFeedOn) cap >> output;

	Mat BW_image = Mat::zeros(Size(output.cols, output.rows), CV_8UC1);
	uint8_3_bgr_to_8_bit_gray(output, BW_image);
	image_1 = BW_image.clone();
	image_2 = BW_image.clone();

	int x_res = BW_image.cols; //Variables for storing the resolution of the image
	int y_res = BW_image.rows;

	int delta_x = (int)floor(x_res / x_bin_n); //The resolution of each rectangle for HS-corner calculation
	int delta_y = (int)floor(y_res / y_bin_n);
	int half_delta_x = (delta_x - 1) / 2; //For odd values of delta
	int half_delta_y = (delta_y - 1) / 2;

	cout << "deltax, y:  " << to_string(delta_x) << '\t' << to_string(delta_y) << '\n';

	Mat binary_image_mat = Mat::zeros(Size(x_res, y_res), CV_8UC1);
	Mat zero_boi = binary_image_mat.clone();

	clock_t start, end;

	featurePoint list[NMAX], list_Old[NMAX];

	//Init parameters for stereo matching and motion estimation
	Mat H_Trans = (Mat_<double>(4, 4) <<1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);// homogenius transformation matrix
	//Mat onesMat = Mat::ones(1, 100, CV_64F);
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

	int i = 0;//image number
	while (1) {
		//cap >> output;
		i++;
		if (i == imageArray_L.size()) break;
		//output = imageArray_L[i];
		//if(LiveFeedOn) cap >> output;
		//cv::resize(imageArray_L[i], output, cv::Size(), 0.25, 0.25);
		output = imageArray_L[i];

		if (!useOpenCV) {//------------HOMEMADE FEATURE DETACTION AND TRACKING----------------
			//uint8_3_bgr_to_8_bit_gray(output, BW_image);
			GaussianBlur(output, image_1, Size(5, 5), 0); //Totally mandatory in the precense of noise

			//Finds good corner features using FAST and puts the coordinates into list_Old
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
		
		
	
		//------stereo Matching-------------------------------------
		if (!useOpenCV) {
			//if(_3D_points_prev.cols) > 0)
			_3D_points.release();
			_3D_points_prev.release();
			disp_l.release();
			disp_r.release();
			_3D_points = Mat_<double>(3, 0);
			_3D_points_prev = Mat_<double>(3, 0);
			GaussianBlur(imageArray_L[i], imageArray_L[i], Size(5, 5), 0);
			GaussianBlur(imageArray_R[i], imageArray_R[i], Size(5, 5), 0);
			disp_l = imageArray_L[i].clone();
			disp_r = imageArray_R[i].clone();
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
					if (z > 0 && z < 10 && z_prev>0 && z_prev < 10 && abs(z - z_prev) < 1) 
					{
						//cout << "\n x , y =" + to_string(ul) + " , " + to_string(vl);
						//cout << "\nDepth for idx= " + to_string(i) + " x y z = " + to_string(x) + " " + to_string(y) + " " + to_string(z) + " ";
						hconcat(_3D_points, currentPoint, _3D_points); //puts all the points into a matrix, each colum=[x;y;z]
						hconcat(_3D_points_prev, prevPoint, _3D_points_prev); //puts all the points into a matrix, each colum=[x;y;z]
					}
				}
			}
		}
		start = clock();
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
						//cout << "\n x , y =" + to_string(ul) + " , " + to_string(vl);
						//cout << "\nDepth for idx= " + to_string(i) + " x y z = " + to_string(x) + " " + to_string(y) + " " + to_string(z) + " ";
						hconcat(_3D_points, currentPoint, _3D_points); //puts all the points into a matrix, each colum=[x;y;z]
						hconcat(_3D_points_prev, prevPoint, _3D_points_prev); //puts all the points into a matrix, each colum=[x;y;z]
					}
				}
			}
		}

		end = clock();
		double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
		cout << "\n Time: " << time_taken << endl;

		Mat rot;

		if(1){
		//-----Motion Estimation------------------------------------
		//starting_Position
		if (i >= 2) {
			//estimateAffine3D(_3D_points, _3D_points_prev, H_Trans, inliers2, 3.0, 0.99); 

			if (_3D_points.cols and _3D_points_prev.cols) { // Checks if there are points in both samples, else H_Trans is left unchanged.
				//H_Trans = motionEstimationIt(_3D_points, _3D_points_prev);
				kabsch_function(_3D_points_prev, _3D_points, H_Trans, rot);
			}
				
			starting_Position = H_Trans * starting_Position;
			cout << starting_Position;
			hconcat(camera_path, starting_Position, camera_path);

			double alphaA = 0;
			alphaA += (180 / 3.14159) * atan2(H_Trans.at<double>(2, 1), H_Trans.at<double>(2, 2));
			double betaA = 0;
			betaA += (180 / 3.14159)* atan2(-H_Trans.at<double>(2, 0), sqrt(pow(H_Trans.at<double>(2, 1), 2.0) + pow(H_Trans.at<double>(2, 2), 2.0)));
			double gammaA = 0;
			gammaA += (180 / 3.14159)* atan2(H_Trans.at<double>(1, 0), H_Trans.at<double>(0, 0));
			cout << "Angle alpha: " << alphaA << '\n';
			cout << "Angle beta: " << betaA << '\n';
			cout << "Angle gamma: " << gammaA << '\n';

			//Mat imageBoi = Mat::zeros(Size(400, 400), CV_8UC3);
			//plotAngle(imageBoi, alphaA, betaA, gammaA);
			//imshow("Angle plot", imageBoi);

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
	//std::this_thread::sleep_for(std::chrono::seconds(5));
	int n;
	cin >> n;
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
	//data = camaraPath;
	//cv::Range rowsToExtract(0, 3);  // Extract rows from index 0 to 2, (the first 3 rows) 
	//double data = data.rowRange(rowsToExtract);

	// Check if data is of the correct type and size
	//if (camaraPath.empty() || camaraPath.cols != 2 || camaraPath.type() != CV_32F) {
	//	throw std::runtime_error("Input data must be a non-empty cv::Mat with 2 columns and type CV_32F");
	//}

	// Find min and max values for normalization
	
	double minX, maxX, minY, maxY;
	cv::minMaxLoc(camaraPath.col(0), &minX, &maxX);
	cv::minMaxLoc(camaraPath.col(1), &minY, &maxY);
	//int axisScale = 0.5;minX = -1* axisScale; maxX = 1* axisScale;minY = -1* axisScale;maxY = 1* axisScale;

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
	//cv::waitKey(0);
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

		//cv::line(plotImage, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
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

		//char key = (char)cv::waitKey(30);
		//if (key == 27) break;
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
	cout << endl << H_inv << endl;
	//transpose(points_new_avg, points_new_avg);

	//Mat onesMat = Mat::ones(1,points_new_avg.cols, CV_64F);
	//cv::vconcat(points_new_avg,onesMat,points_new_avg);

	//Mat Trans = H_inv * points_new_avg;

	//transpose(points_old_avg, points_old_avg);
	//
	//cv::vconcat(points_old_avg, onesMat, points_old_avg);

	//Mat Trans2 = Trans - points_old_avg;
	//transpose(Trans2, Trans2);

	transpose(points_new_avg, points_new_avg);

	Mat onesMat = Mat::ones(1, points_new_avg.cols, CV_64F);
	cv::vconcat(points_new_avg, onesMat, points_new_avg);
	transpose(points_old_avg, points_old_avg);
	cv::vconcat(points_old_avg, onesMat, points_old_avg);

	//convert points_new_avg from position vector to a homogenous translation matrix
	Mat n_avg = Mat::eye(4, 4, CV_64F);
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
	//cout << points_old_avg;
	Mat Trans = H_out_temp * (n_avg_inv * (H_inv * points_old_avg));

	Mat Trans2;
	transpose(Trans, Trans2);
	//R should probably be treated independently, because the translation is in world coordinates, so rotation should not be done on the position.
	//cout << "Trans = " << endl << " " << Trans << endl << endl;
	//cout << "Trans2 = " << endl << " " << Trans2 << endl << endl;
/*
	Mat ROTATION = (Mat_<double>(4, 4) <<
		R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), 0,
		R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), 0,
		R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), 0,
		0.0,				0.0,				0.0,				1.0);*/
	Mat TRANSLATION = (Mat_<double>(4, 4) <<
		1.0, 0.0, 0.0, Trans2.at<double>(0, 0),
		0.0, 1.0, 0.0, Trans2.at<double>(0, 1),
		0.0, 0.0, 1.0, Trans2.at<double>(0, 2),
		0.0, 0.0, 0.0, 1.0);

	TRANSLATION.copyTo(Tran);
	H_out_temp.copyTo(Rot);

}



void plotAngle(Mat dst, double alpha, double beta, double gamma) {
	int rows = dst.rows;
	int cols = dst.cols;
	int x1, x2, y1, y2;
	double lort = 3.1 / 180;
		alpha *= lort;
		beta *= lort;
		gamma *= lort;
	Scalar_<int> red = Scalar(255, 0, 0);
	Scalar_<int> green = Scalar(0, 255, 0);
	Scalar_<int> blue = Scalar(0, 0, 255);

	int length = 30;
	for (int i = 0; i < 1; i++) {
		x1 = 1 * (rows / 4);
		y1 = (rows / 2);
		x2 = x1+cos(alpha) * length;
		y2 = y1+sin(alpha) * length;
		arrowedLine(dst, Point(x1, dst.rows - 1 - y1), Point(x2, dst.rows - 1 - y2), red, 2);
	}    for (int i = 1; i < 2; i++) {
		x1 = 2 * (rows / 4);
		y1 = (rows / 2);
		x2 = x1+cos(beta) * length;
		y2 = y1+sin(beta) * length;
		arrowedLine(dst, Point(x1, dst.rows - 1 - y1), Point(x2, dst.rows - 1 - y2), green, 2);
	}    for (int i = 2; i < 3; i++) {
		x1 = 3 * (rows / 4);
		y1 = (rows / 2);
		x2 = x1+cos(gamma) * length;
		y2 = y1+sin(gamma) * length;
		arrowedLine(dst, Point(x1, dst.rows - 1 - y1), Point(x2, dst.rows - 1 - y2), blue, 2);
	}
}