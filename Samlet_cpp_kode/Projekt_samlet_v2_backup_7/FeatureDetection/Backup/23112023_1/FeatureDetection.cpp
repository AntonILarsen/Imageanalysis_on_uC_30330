#include "FeatureDetection.h"

// For the sake of the other guys, we must use an odd X odd kernel size. We shall return the top left coordinate of the kernel plus (kernel size - 1)/2
const int x_bin_n = 45; //Variables for setting the number f x and y bins in which
const int y_bin_n = 34; //we find the RMS value to judge feasibility of feature recognition

//Can this perhaps be declared in the header file? How do we initialise function with the concrete values of x_bin_n and y_bin_n?
int feature_list(Mat input_image, int output_list[2][x_bin_n * y_bin_n], int threshold);




int main() {

	Mat output;
	Mat image_1;
	Mat image_2;
	VideoCapture cap(0); //Try changing to 1 if not working
	//Check if camera is available
	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return 0;
	}

	cap.set(CAP_PROP_EXPOSURE, -5.0);
	cap.set(CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CAP_PROP_FRAME_HEIGHT, 240);
	
	cout << "Exposure value: " << to_string(cap.get(CAP_PROP_EXPOSURE)) << '\n';
	cout << "Gain value: " << to_string(cap.get(CAP_PROP_GAIN)) << '\n';

	namedWindow("Feature image", WINDOW_KEEPRATIO); //Create a new window with the name "RMS image"

	cap >> output;

	Mat BW_image = Mat::zeros(Size(output.cols, output.rows), CV_8UC1);
	uint8_3_bgr_to_8_bit_gray(output, BW_image);
	image_1 = BW_image.clone();
	image_2 = BW_image.clone();

	int x_res = BW_image.cols; //Variables for storing the resolution of the image
	int y_res = BW_image.rows;

	int delta_x = (int)floor(x_res / x_bin_n); //The resolution of each rectangle for HS-corner calculation
	int delta_y = (int)floor(y_res / y_bin_n);

	cout << "deltax, y:  " << to_string(delta_x) << '\t' << to_string(delta_y) << '\n';

	Mat binary_image_mat = Mat::zeros(Size(x_bin_n, y_bin_n), CV_8UC1);
	Mat feature_image = Mat::zeros(Size(x_bin_n, y_bin_n), CV_8UC1);

	clock_t start, end;

	featurePoint list[NMAX], list_Old[NMAX];

	std::vector<cv::KeyPoint> keypoints_fast, keypoints_fast_old;
	cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(20);

	feature_detection(BW_image, list, 0.1);

	image_1 = BW_image.clone();
	fast->detect(BW_image, keypoints_fast_old);
	int sizeThing_old = size(keypoints_fast_old);

	image_2 = BW_image.clone();
	fast->detect(BW_image, keypoints_fast);
	int sizeThing = size(keypoints_fast);

	//Show Camera output
	while (1) {
		cap >> output;

		uint8_3_bgr_to_8_bit_gray(output, BW_image);
		GaussianBlur(BW_image, BW_image, Size(5, 5), 0);
		

		/*feature_image = Mat::zeros(Size(x_bin_n, y_bin_n), CV_8UC1);

		int feat_n = 0;
		while (feat_n < NMAX)
		{
			if (list[feat_n].active)
			{
				feature_image.at<uchar>(Point((list[feat_n].x - (delta_x - 1)/2)/delta_x, (list[feat_n].y - (delta_y - 1) / 2) /delta_y)) = 255;
			}
			feat_n++;
		}*/

		for (size_t i = 0; i < sizeThing; ++i) {
			//BW_image.at<uchar>(Point(keypoints_fast[i].pt.x, keypoints_fast[i].pt.y)) = 255;
			//cout << "Size: " << sizeThing << '\n';
			//std::cout << "FAST Keypoint #:" << i;
			//std::cout << " Pt " << keypoints_fast[i].pt << " Angle " << keypoints_fast[i].angle << " Response " << keypoints_fast[i].response << " Octave " << keypoints_fast[i].octave << std::endl;
		}

		// Calculating total time taken by the program.
		


		imshow("Feature image", feature_image);
		imshow("Camera image", BW_image);

		char c = (char)waitKey(10);
		if (c == 27) break; //Press escape to stop program
		if (c == 'a')
		{
			for (int i = 0; i < NMAX; i++)
				list[i].active = false;
		}


		image_1 = BW_image.clone();
		fast -> detect(image_1, keypoints_fast_old);
		sizeThing_old = size(keypoints_fast_old);

		start = clock();

		int score = 0;
		int old_Score = INT_MAX;

		for (int i = 0; i < NMAX; i++)
		{
			list[i].active = false;
			list_Old[i].active = false;
		}

		for (int ne = 0; ne < sizeThing && (sizeThing != 0) && (sizeThing_old != 0); ne++) //Iterates through the new list of FAST features
		{
			int blablaIndex = 0;
			old_Score = INT_MAX;
			for (int ol = 0; ol < sizeThing_old && (ol < NMAX); ol++) //Iterates through the old list of FAST features
			{
				int position_x = (int)keypoints_fast[ne].pt.x;
				int position_y = (int)keypoints_fast[ne].pt.y;
				int position_x_old = (int)keypoints_fast_old[ol].pt.x;
				int position_y_old = (int)keypoints_fast_old[ol].pt.y;

				if ((position_x - delta_x > 0) && (position_x + delta_x < x_res)
					&& (position_y - delta_y > 0) && (position_y + delta_y < y_res)
					&& (position_x_old - delta_x > 0) && (position_x_old + delta_x < x_res)
					&& (position_y_old - delta_y > 0) && (position_y_old + delta_y < y_res))
				{
					score = 0;

					for (int x = 0; x < delta_x; x++)
					{
						for (int y = 0; y < delta_y; y++)
						{
							score += (int)abs(image_1.at<uchar>(Point(position_x_old + x, position_y_old + y))
								- image_2.at<uchar>(Point(position_x + x, position_y + y)));
						}
					}

					if (score < old_Score && ne < NMAX)
					{
						old_Score = score;
						blablaIndex = ol;
					}
				}


			}
			//cout << "Index: " << blablaIndex << "  Score:" << old_Score << '\n';
			list[blablaIndex].x = keypoints_fast[ne].pt.x;
			list[blablaIndex].y = keypoints_fast[ne].pt.y;
			list[blablaIndex].active = (old_Score < 300) ? true : false;
			list[blablaIndex].score = old_Score;

			list_Old[blablaIndex].x = keypoints_fast_old[blablaIndex].pt.x;
			list_Old[blablaIndex].y = keypoints_fast_old[blablaIndex].pt.y;
			list_Old[blablaIndex].active = (old_Score < 300) ? true : false;
			list_Old[blablaIndex].score = old_Score;


		}

		image_2 = BW_image.clone();
		fast->detect(image_2, keypoints_fast);
		sizeThing = size(keypoints_fast);

		end = clock();

		plotTracking(image_2, image_1, list, list_Old);

		double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
		cout << "Time taken by program is : " << fixed
			<< time_taken << setprecision(5);
		cout << " sec " << endl;
	}



/*  //3D transformation sketch

	Mat Points = (Mat_<double>(12, 3) << -7.5000, -5.0000, -0.2320, -7.5000, 0, 0.9602, -7.5000, 5.0000, -0.8787, -2.5000, -5.0000, 0.1188, -2.5000, 0, 0.1188, -2.5000, 5.0000, 0.1188, 2.5000, -5.0000, 0.4695, 2.5000, 0, -0.7227, 2.5000, 5.0000, 1.1163, 7.5000, -5.0000, -0.5382, 7.5000, 0, -0.7905, 7.5000, 5.0000, 0.2599);
	Mat PointsTranspose = Mat::zeros(Size(Points.cols, Points.rows), CV_64F);

	transpose(Points, PointsTranspose);

	Mat PointsTransformed = (Mat_<double>(12, 3) << -7.5000, -4.6836, -1.7657, -7.5000, -0.2967, 0.9132, -7.5000, 5.0268, 0.7094, -2.5000, -4.7920, -1.4321, -2.5000, -0.0367, 0.1130, -2.5000, 4.7186, 1.6580, 2.5000, -4.9004, -1.0985, 2.5000, 0.2233, -0.6873, 2.5000, 4.4103, 2.6067, 7.5000, -4.5890, -2.0570, 7.5000, 0.2443, -0.7518, 7.5000, 4.6750, 1.7922);
	cout << "Points = " << endl << " " << Points << endl << endl;
	cout << "PointsTransformed = " << endl << " " << PointsTransformed << endl << endl;
	cout << "PointTranspose = " << endl << " " << PointsTranspose << endl << endl;

	Mat U = Mat::zeros(Size(3, 3), CV_64F);
	Mat U_transpose = U.clone();
	Mat V = U.clone();
	Mat S = U.clone();

	Mat H = PointsTranspose * PointsTransformed;
	cout << "H = " << endl << " " << H << endl << endl;

	SVD::compute(H, S, U, V);

	double d = - 1.0 + 2 * (determinant(U) > 0.0); //Signum function
	Mat signumMatrix = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, d);

	cout << "U = " << endl << " " << U << endl << endl;
	cout << "V = " << endl << " " << V << endl << endl;
	cout << "S = " << endl << " " << S << endl << endl;
	cout << "d = " << endl << " " << d << endl << endl;


	transpose(U, U_transpose);

	Mat R = V * signumMatrix * U_transpose;

	cout << "R = " << endl << " " << R << endl << endl;
	


	Mat testBoi = (Mat_<double>(3, 3) << 10, 2, 1, 6, 23, 5, 2, 4, 7);
	Mat sqrtBoi = testBoi.clone();

	matrix_square_root(testBoi, sqrtBoi);
	cout << "sqrtBoi = " << endl << " " << sqrtBoi << endl << endl;
*/


	destroyAllWindows();

	return 0;
}

int feature_compare(Mat image_old, Mat image_latest, featurePoint list_old[NMAX], featurePoint list_latest[NMAX], int threshold)
{
	return 0;
}


int feature_detection2(Mat image, featurePoint list[NMAX], double threshold)
{
	std::vector<cv::KeyPoint> keypoints_fast;
	cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(threshold);

	fast -> detect(image, keypoints_fast);
	int sizeOfList = size(keypoints_fast);

	for (int i = 0; i < sizeOfList && (i < NMAX); i++) //Iterates through the old list of FAST features
	{
		list[i].x = (int)keypoints_fast[i].pt.x;
		list[i].y = (int)keypoints_fast[i].pt.y;
	}

	return 0;
}

int feature_detection(Mat latest_image, featurePoint list[NMAX], double threshold)
{
	int x_res = latest_image.cols;
	int y_res = latest_image.rows;

	int list_length = 0; //This is found later in this function. The number of currently active features for tracking

	int feature_array[2][x_bin_n * y_bin_n]; //Defining the list of where in the picture good features are
	int feature_array_length = 0; //The number of found features for potential tracking. Will often be reduced to NMAX

	int start_x = 0; //The starting coordinates for the current Harris-Stephenson window
	int start_y = 0;

	int delta_x = (int)floor(x_res / x_bin_n); //The resolution of each rectangle for HS-corner calculation
	int delta_y = (int)floor(y_res / y_bin_n);
	int half_delta_x = (delta_x - 1) / 2;
	int half_delta_y = (delta_y - 1) / 2;

	int corner_val = 0; //Variable for storing the result of corner detection for each block/window

	Mat subset_image = Mat::zeros(Size(delta_x, delta_y), CV_8UC1);

	//Loop selecting these blocks/windows
	for (int j = 0; j < y_bin_n; j++) {
		for (int i = 0; i < x_bin_n; i++) {
			start_x = (i * x_res) / x_bin_n;
			start_y = (j * y_res) / y_bin_n;

			//Copy the current rectangle to the subset_image
			image_subset_copy(latest_image, subset_image, start_x, start_y, delta_x, delta_y); //15 ms execution time with 640x480 with 80x60 of 8x8 bins

			corner_val = (int)min(max(HS_window_classifier2(subset_image, threshold) * 0.0000005, 0.0), 255.0);

			//If a good feature is found, put its coordinates into feature array
			if (corner_val >= 32)
			{
				feature_array[0][feature_array_length] = i;
				feature_array[1][feature_array_length] = j;
				feature_array_length++;
			}
		}
	}

	//If we have more valid feature areas than our maximum limit (NMAX), we go ahead and sample random features until we have NMAX features
	//We use uniform sampling in x and y. It would probably be better to use blue noise to avoid feature clustering, but so far we're using uniform noise.
	//This approach has the potential to sample the same feature twice

	int i = 0;

	while (i < NMAX)
	{

		if (list[i].active) //If the current index in the list of tracked features is 'active', it is currently being tracked, and shouldn't be updated
			list_length++; //We count up how many active items we have in the list

		else if (feature_array_length > 1)	//If the current index is not being tracked/active, we put the 'new' feature point coordinates into that index. 
		{									//Also, there must actually be any features to track
			int random_ind = rand() % (feature_array_length - 1); //Generate a random index for use of our list of indices of corner features

			bool alreadyExisting = false; //A flag used when a new potential feature point is already existing / too close to an existing feaure point

			//Where the randomly selected new corner feature is located in global image coordinates. Here using the middle of the kernel as the saved location
			int newFeaturePosition_x = (feature_array[0][random_ind] * x_res) / x_bin_n + half_delta_x;
			int newFeaturePosition_y = (feature_array[1][random_ind] * y_res) / y_bin_n + half_delta_y;

			for (int n = 0; (n < NMAX) && (!alreadyExisting); n++)
			{
				//Checking that the new feature position is at least one kernel size away from an existing feature in all points
				if ((abs(newFeaturePosition_x - list[n].x) < delta_x) && (abs(newFeaturePosition_y - list[n].y) < delta_y) && list[n].active)
					alreadyExisting = true; //If the feature is active and isn't far enough away, we just end the for loop and set this flag high
			}

			if (!alreadyExisting) //If the found feature point wasn't already in the list, we can save it to the list. 
			{ 
				// It is possible, although not likely, that all the random guesses we make are already existing feature points
				// In that case, we wouldn't be adding new points to the list. We won't keep on guessing until we get enough (new) feature points,
				// as that could potentially last forever

				list[i].x = newFeaturePosition_x; 
				list[i].y = newFeaturePosition_y;
				list[i].active = true;
				list_length++;

				//Same as above, but using the top left of the kernel as the saved location. Same same, but different
				//list[i].x = (feature_array[0][random_ind] * x_res) / x_bin_n; //use the same indices for both
				//list[i].y = (feature_array[1][random_ind] * y_res) / y_bin_n; //use the same indices for both. We're acessing one random COORDINATE, which is stored in two array indices for x and y
				//list[i].active = true;
			}
		}

		i++;
	}
	
	int returnVal = 0;

	if (list_length < NMAX)
		returnVal = 1;

	//REMEMBER TO ADD A FEATURE TO CHECK IF THE CURRENT FOUND FEATURE AREA IS TOO CLOSE TO AN EXISTING ONE

	return returnVal;

}

void matrix_square_root(const cv::Mat& A, cv::Mat& sqrtA)
{	//Stolen from https://cppsecrets.com/users/140211510511010310497110117981049711848494864103109971051084699111109/C00-OpenCV-cvsqrt.php
	cv::Mat U, V, Vi, E;
	cv::eigen(A, E, U);
	V = U.clone();
	cv::invert(V, Vi);
	cv::sqrt(E, E);         //We have to assume that A is positively defined,                         
	// otherwise its square root will be complex-valued                       
	sqrtA = V * Mat::diag(E) * Vi; //element wise square root of an array.
}


double HS_window_classifier2(Mat input_image, float k)
{
	int imseg_x = input_image.cols;
	int imseg_y = input_image.rows;

	long int diff_x = 0;
	long int diff_y = 0;
	long long int harris_descriptor[2][2] = { {0,0}, {0,0} }; //It shouldn't be necessary to use a long long here. Why is that?
	long long int det = 0;
	int currentSobel_x = 0;
	int currentSobel_y = 0;

	for (int j = 1; j < imseg_y - 1; j++)
	{
		for (int i = 1; i < imseg_x - 1; i++)
		{
			currentSobel_x	= (input_image.at<uchar>(Point(i - 1, j - 1)) + input_image.at<uchar>(Point(i - 1, j + 1)))
							+ 2 * input_image.at<uchar>(Point(i - 1, j))
							- (input_image.at<uchar>(Point(i + 1, j - 1)) + input_image.at<uchar>(Point(i + 1, j + 1)))
							- 2 * input_image.at<uchar>(Point(i + 1, j));

			currentSobel_y	= (input_image.at<uchar>(Point(i - 1, j - 1)) + input_image.at<uchar>(Point(i + 1, j - 1)))
							+ 2 * input_image.at<uchar>(Point(i, j - 1))
							- (input_image.at<uchar>(Point(i - 1, j + 1)) + input_image.at<uchar>(Point(i + 1, j + 1)))
							- 2 * input_image.at<uchar>(Point(i, j + 1));				

			diff_x = -(long)currentSobel_x;
			diff_y = -(long)currentSobel_y;

			harris_descriptor[0][0] += diff_x * diff_x;
			harris_descriptor[0][1] += diff_x * diff_y;
			harris_descriptor[1][0] += diff_x * diff_y;
			harris_descriptor[1][1] += diff_y * diff_y;
		}
	}

	det = harris_descriptor[0][0] * harris_descriptor[1][1] - harris_descriptor[0][1] * harris_descriptor[1][0]; //'Overflow' or something the like occurs here if harris_des. is not long long

	return ((double)det - k * ((double)harris_descriptor[0][0] + (double)harris_descriptor[1][1])*((double)harris_descriptor[0][0] + (double)harris_descriptor[1][1]));

}

double HS_window_classifier(Mat input_image, double k)
{
	int imseg_x = input_image.cols;
	int imseg_y = input_image.rows;

	double diff_x = 0;
	double diff_y = 0;
	double harris_descriptor[2][2] = {{0,0}, {0,0}};
	double det = 0;

	Mat diffX_image = Mat::zeros(Size(imseg_x, imseg_y), CV_64F);
	Mat diffY_image = Mat::zeros(Size(imseg_x, imseg_y), CV_64F);

	Sobel(input_image, diffX_image, CV_64F, 1, 0);
	Sobel(input_image, diffY_image, CV_64F, 0, 1);

	for(int j = 1; j < imseg_y - 1; j++)
	{
		for (int i = 1; i < imseg_x - 1; i++)
		{
			diff_x = diffX_image.at<double>(Point(i, j));
			diff_y = diffY_image.at<double>(Point(i, j));

			harris_descriptor[0][0] += diff_x * diff_x;
			harris_descriptor[0][1] += diff_x * diff_y;
			harris_descriptor[1][0] += diff_x * diff_y;
			harris_descriptor[1][1] += diff_y * diff_y;
		}
	}

	det = harris_descriptor[0][0] * harris_descriptor[1][1] - harris_descriptor[0][1] * harris_descriptor[1][0];
	//cout << "DET: " << to_string(det) << "  HS10: " << to_string(harris_descriptor[1][0]) << "\n";

	return (det - k * pow(harris_descriptor[0][0] + harris_descriptor[1][1], 2));

}

//Find the non-zero pixels in an uint8 image and puts these indices into output_list. output_list must have dims [2][x_res * y_res]
int feature_list(Mat input_image, int output_list[2][x_bin_n*y_bin_n], int threshold)
{
	int x_res = input_image.cols;
	int y_res = input_image.rows;

	int list_length = 0;

	for (int y = 0; y < y_res; y++)
	{
		for (int x = 0; x < x_res; x++)
		{
			if (input_image.at<uchar>(Point(x, y)) >= threshold)
			{
				output_list[0][list_length] = x;
				output_list[1][list_length] = y;
				list_length++;
			}
			else
			{
				output_list[0][list_length] = 0;
				output_list[1][list_length] = 0;
			}
		}
	}

	return list_length;
}

double image_sum(Mat input_image)
{
	
	int x_res = input_image.cols;
	int y_res = input_image.rows;

	double pixelSum = 0;

	for (int y = 0; y < y_res; y++)
	{
		for (int x = 0; x < x_res; x++)
		{
			pixelSum += input_image.at<uchar>(Point(x, y));
		}
	}

	return pixelSum;
}

double RMS_value(Mat input_image, bool mean_subtract) 
{
	double RMSval = 0;

	int x_res = input_image.cols;
	int y_res = input_image.rows;
	double image_area = ((double)x_res) * ((double)y_res);
	double square = 0;
	double avg = 0;

	if (mean_subtract)
	{
		double mean = image_sum(input_image) / image_area;

		for (int y = 0; y < y_res; y++)
		{
			for (int x = 0; x < x_res; x++)
			{
				square = ((double)input_image.at<uchar>(Point(x, y)) - mean) * ((double)input_image.at<uchar>(Point(x, y)) - mean); //We use pixel*pixel to square as it is a bit faster than pow(pixel,2)
				avg += square; //Instead of dividing a potentially huge number in the end, it is just done along the way. It is a linear operation, so it is possible to do
			}
		}
	} 
	else
	{
		for (int y = 0; y < y_res; y++)
		{
			for (int x = 0; x < x_res; x++)
			{
				square = ((double)input_image.at<uchar>(Point(x, y))) * ((double)input_image.at<uchar>(Point(x, y))); //We use pixel*pixel to square as it is a bit faster than pow(pixel,2)
				avg += square; //Instead of dividing a potentially huge number in the end, it is just done along the way. It is a linear operation, so it is possible to do
			}
		}
	}

	RMSval = sqrt(avg / image_area);

	return RMSval;
}

//This function takes a bigger input image and copies a subset of its content (a rectangle) to another image (output image)
void image_subset_copy(Mat input_image, Mat output_image, int start_x, int start_y, int delta_x, int delta_y)
{
	//It requires the output image to be big enough to contain the subset of data from the input image.
	// 
	//This function also requires that the start_x + delta_x OR start_y + delta_y DOES NOT exceed the array boundaries of input_image

	int end_x = start_x + delta_x; //The end of the subset area of the input image is precalculated
	int end_y = start_y + delta_y;

	for (int y = start_y; y < end_y; y++)
	{
		for (int x = start_x; x < end_x; x++)
		{
			//The indexing to the output image is subtracted start_x and start_y, respectively, as to index with just deltaX and deltaY
			output_image.at<uchar>(Point(x-start_x, y-start_y)) = input_image.at<uchar>(Point(x, y));
		}
	}
}

//This function requires that the start_x + delta_x OR start_y + delta_y DOES NOT exceed image array boundaries
double image_sum_subarea(Mat input_image, int start_x, int start_y, int delta_x, int delta_y)
{

	double pixelSum = 0; //The sum of the pixels to be summed is reset

	int end_x = start_x + delta_x; //The end of the summing area position(s) is precalculated
	int end_y = start_y + delta_y;

	for (int y = start_y; y < end_y; y++)
	{
		for (int x = start_x; x < end_x; x++)
		{
			pixelSum += input_image.at<uchar>(Point(x, y));
		}
	}

	return pixelSum;
}


void uint8_3_bgr_to_8_bit_gray(Mat &input_img, Mat &output_img)
{
	//We take the input and output images as pointers
	//and just edit them in place

	vector<Mat> channels(3); //Define split channels

	split(input_img, channels); //Split input image into channels

	//BGR format of weighting 0.3R, 0.6G, 0.11B
	output_img =  0.11*channels[0] + 0.6 * channels[1] + 0.3 * channels[2];
}

void uint8_3_bgr_to_8_bit_gray_2(Mat& input_img, Mat& output_img, int channel_no)
{
	//We take the input and output images as pointers
	//and just edit them in place

	vector<Mat> channels(3); //Define split channels

	split(input_img, channels); //Split input image into channels

	output_img = channels[channel_no];
}

void histogram_boi_anim(Mat& input_img, char channel_no, string window_name, bool normalise)
{
	int k = 256;
	unsigned char value = 0; // index value for the histogram
	int histogram[256]; // histogram array - remember to set to zero initially

	while (k-- > 0)
		histogram[k] = 0; // reset histogram entry

	int width = input_img.cols;
	int height = input_img.rows;

	int maxN = 1;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			value = input_img.at<Vec3b>(i, j)[channel_no];
			histogram[value] += 1;
		}
	}

	for (int i = 255; i >= 0; i--)
	{
		if (histogram[i] > maxN)
			maxN = histogram[i];
	}

	if (!normalise)
		maxN = width * height;

	k = 256;
	int hist_height = (int)(0.7 * k);
	//Create an empty image for the histogram of size k, k*0.5
	Mat hist_img = Mat::zeros(Size(k, hist_height), CV_8UC1);

	for (int i = 0; i < hist_height; i++)
	{
		for (int j = 0; j < k; j++)
		{
			if (i > (hist_height * (histogram[j] - 1)) / maxN) //Normalised wrt. maximum of histogram
				hist_img.at<uchar>(hist_height - 1 - i, j) = 255; //Why the f is x and y flipped here???
		}
	}

	imshow(window_name, hist_img);
}


void SAD(cv::Mat ref, cv::Mat input, featurePoint list[NMAX], int ksize, int dist) {
	//ksize is hard coded for now:
	ksize = 7;

	// Sum-of-Absolute-Differences. 
	// For each active feature point in list[NMAX]:
	// Samples a ksize*ksize subimage of "ref" centered on the coordinate of the feature point.
	// then takes SAD for every pixel in a "dist" (x + / - dist, y +/- dist) distance from the feature point coordinate.
	// it compares each previous SAD to the next one and saves the lowest one.
	// NB this function returns nothing because it changes the featurePoint list[NMAX] entries directly: remember to back it up.


	// Assuming uneven ksize (ex. ksize = 7 gives kside = 3)
	int kside = ksize / 2;

	// val_th sets the threshold for a good-enough match (matches are ony considered matches if the SAD is less than val_th)

	int val_th = 300000;



	for (int index = 0; index < NMAX; index++) {

		cout << "Index: " << index << " of " << NMAX << '\n';

		//val is used for the summing
		int val = 0;

		//prevval is used to compare the new SAD to the previous in each iteration (every pixel)
		//it is initialized to the maximum possible int so that val will always be smaller in the first check
		int prevval = INT_MAX;
		;
		//load the coordinate of the feature point into some variables before we overide them in the list[NMAX]
		int sample_x = list[index].x;
		int sample_y = list[index].y;

		//get the limits for the for loops
		int xmin = max(kside, sample_x - dist);
		int xmax = min(input.cols - kside, sample_x + dist);
		int ymin = max(kside, sample_y - dist);
		int ymax = min(input.rows - kside, sample_y + dist);
		int sample_x_offset = sample_x - kside;
		int sample_y_offset = sample_y - kside;
		int ref_x_offset;
		int ref_y_offset;


		int th = 0;


		// Only run the algo for active points
		if (list[index].active)
		{
			for (int k = 0; k < ksize; k++)
			{
				for (int l = 0; l < ksize; l++)
				{
					//Sum of absolute differences 
					th += (int)ref.at<uchar>(Point(sample_x_offset + k, sample_y_offset + l));
				}
			}
			val_th = th - ksize*ksize*8;

			for (int i = xmin; i < xmax; i++)
			{
				ref_x_offset = i - kside;

				for (int j = ymin; j < ymax; j++)
				{
					ref_y_offset = j - kside;
					//Reset the sum and apply kernel over a window of ksize*ksize centered around coordinate (i, j) = (x, y)
					val = 0;
					for (int k = 0; k < ksize; k++)
					{
						for (int l = 0; l < ksize; l++)
						{
							//Sum of absolute differences 

							val += abs(ref.at<uchar>(Point(sample_x_offset + k, sample_y_offset + l)) - input.at<uchar>(Point(ref_x_offset + k, ref_y_offset + l)));
						}
					}
					if (val < prevval)
					{
						//change the entry if match is better than previous
						//sets a new coordinate together with the score = val and keeps it active = 1
						list[index] = { i, j, (int)val, 1 };

						//store previous val
						prevval = val;
					}
				}
			}

			if (val > val_th) {
				//Deactivate point if match is not good enough (val_th needs to be adjusted manually)
				list[index].active = 0;
			}
		}
	}
	return;
}


void plotTracking(cv::Mat cur_img, cv::Mat prev_img, featurePoint cur_list[NMAX], featurePoint prev_list[NMAX]) {

	//Init output image
	cv::Mat output_img;
	cv::Mat output_single;
	addWeighted(cur_img, 0.5, prev_img, 0.5, 0.0, output_single);
	//Concatenate the current and previous image so they can be shown side by side
	cv::hconcat(cur_img, prev_img, output_img);

	//Convert back to color to be able to draw colored "tracking lines"
	cv::cvtColor(output_img, output_img, COLOR_GRAY2BGR);
	cv::cvtColor(output_single, output_single, COLOR_GRAY2BGR);

	//Iterate through the current feature point list and plot the points that are currently active 
	// (which means they where also active in the previous list, since the SAD function can only deactivate points 
	for (int i = 0; i < NMAX; i++) {
		if (cur_list[i].active)
			line(output_img, Point(prev_img.cols + prev_list[i].x, prev_list[i].y), Point(cur_list[i].x, cur_list[i].y), Scalar(uchar(rand() % 256), uchar(rand() % 256), uchar(rand() % 256)), 2, 8, 0);
	}

	for (int i = 0; i < NMAX; i++) {
		if (cur_list[i].active) {
			Scalar color = Scalar(min(cur_list[i].score / 3, 255), min(cur_list[i].score / 3, 255), 0);
			line(output_single, Point(prev_list[i].x, prev_list[i].y), Point(cur_list[i].x, cur_list[i].y), color, 2, 8, 0);
			cv::putText(output_single, //target image
				std::to_string(cur_list[i].score), //text
				cv::Point(cur_list[i].x, cur_list[i].y), //top-left position
				cv::FONT_HERSHEY_SIMPLEX,
				0.2,
				color, //font color
				1);
		}
	}

	//Show the plot
	cv::namedWindow("Tracking", WINDOW_NORMAL);
	cv::namedWindow("Tracking_single", WINDOW_NORMAL);
	cv::imshow("Tracking", output_img);
	cv::imshow("Tracking_single", output_single);

	return;

}







