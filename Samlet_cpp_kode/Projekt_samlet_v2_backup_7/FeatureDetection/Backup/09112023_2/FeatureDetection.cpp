#include "FeatureDetection.h"

// For the sake of the other guys, we must use an odd X odd kernel size. We shall return the top left coordinate of the kernel plus (kernel size - 1)/2
const int x_bin_n = 91; //Variables for setting the number f x and y bins in which
const int y_bin_n = 68; //we find the RMS value to judge feasibility of feature recognition

//Can this perhaps be declared in the header file? How do we initialise function with the concrete values of x_bin_n and y_bin_n?
int feature_list(Mat input_image, int output_list[2][x_bin_n * y_bin_n], int threshold);

int main() {

	cout << "Hello World!";

	Mat output;
	VideoCapture cap(0); //Try changing to 1 if not working
	//Check if camera is available
	if (!cap.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		return 0;
	}

	cap.set(CAP_PROP_EXPOSURE, -7.0);
	cap.set(CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CAP_PROP_FRAME_HEIGHT, 480);
	
	cout << "Exposure value: " << to_string(cap.get(CAP_PROP_EXPOSURE)) << '\n';
	cout << "Gain value: " << to_string(cap.get(CAP_PROP_GAIN)) << '\n';

	namedWindow("RMS image", WINDOW_KEEPRATIO); //Create a new window with the name "RMS image"
	//namedWindow("Binary image", WINDOW_KEEPRATIO); //Create a new window with the name "RMS image"
	namedWindow("Feature image", WINDOW_KEEPRATIO); //Create a new window with the name "RMS image"

	cap >> output;

	Mat BW_image = Mat::zeros(Size(output.cols, output.rows), CV_8UC1);

	int x_res = BW_image.cols; //Variables for storing the resolution of the image
	int y_res = BW_image.rows;


	int start_x = 0; //The starting point (top left) of each rectangle for RMS calculation
	int start_y = 0;
	int delta_x = (int)floor(x_res / x_bin_n); //The resolution of each rectangle for HS-corner calculation
	int delta_y = (int)floor(y_res / y_bin_n);

	cout << "deltax, y:  " << to_string(delta_x) << '\t' << to_string(delta_y) << '\n';

	Mat RMS_image = Mat::zeros(Size(x_bin_n, y_bin_n), CV_8UC1);
	Mat binary_image_mat = Mat::zeros(Size(x_bin_n, y_bin_n), CV_8UC1);
	Mat feature_image = Mat::zeros(Size(output.cols, output.rows), CV_8UC1);
	Mat subset_image = Mat::zeros(Size(delta_x, delta_y), CV_8UC1);

	Mat yet_another = Mat::zeros(Size(x_bin_n, y_bin_n), CV_8UC1);
	

	clock_t start, end;

	//Show Camera output
	while (1) {
		cap >> output;
		uint8_3_bgr_to_8_bit_gray(output, BW_image);
		Mat test = BW_image.clone();
		//GaussianBlur(BW_image, test, Size(5,5), 0);
		BW_image = test;
		resize(BW_image, yet_another, Size(), 0.125, 0.125);

		x_res = BW_image.cols;
		y_res = BW_image.rows;

		imshow("blablabla", BW_image);
		
		start = clock();

		//Loop for selecting the squares for RMS estimation
		for (int j = 0; j < y_bin_n; j++) {
			for (int i = 0; i < x_bin_n; i++) {
				start_x = (i * x_res) / x_bin_n;
				start_y = (j * y_res) / y_bin_n;

				//Copy the current rectangle to the subset_image
				image_subset_copy(BW_image, subset_image, start_x, start_y, delta_x, delta_y); //15 ms execution time with 640x480 with 80x60 of 8x8 bins
				
				RMS_image.at<uchar>(Point(i, j)) = (uchar)min(max(HS_window_classifier2(subset_image, 0.1) * 0.0000005, 0.0), 255.0);// -(uchar)min(max(HS_window_classifier2(subset_image, 0.05) * 0.0000005, 0.0), 255.0);

				//line(BW_image, Point(start_x, start_y), Point(start_x, start_y), 255, 2, LINE_8);
			}
		}

		int feature_array[2][x_bin_n * y_bin_n]; //Defining the list of where in the picture good features are
		int list_length = feature_list(RMS_image, feature_array, 32); //Finding the locations of good features and putting them in a list by means of thresholding

		feature_image = Mat::zeros(Size(x_bin_n, y_bin_n), CV_8UC1);

		const int feat_max = 100;
		int feat_n = 0;

		//If we have more valid feature areas than our maximum limit (feat_max), we go ahead and sample random features until we have feat_max features
		//We use uniform sampling in x and y. It would probably be better to use blue noise to avoid feature clustering, but so far we're using uniform noise.
		if (list_length > feat_max) 
		{
			while (feat_n < feat_max)
			{
				int random_ind = rand() % (list_length - 1);

				feature_image.at<uchar>(Point(feature_array[0][random_ind], feature_array[1][random_ind])) = 255;
				feat_n++;
			}
		}
		//If we have less valid features than our maximum limit (feat_max), we just use all features
		else
		{
			for (int i = 0; i < list_length; i++)
			{
				feature_image.at<uchar>(Point(feature_array[0][i], feature_array[1][i])) = 128;
			}
		}

		//binary_image(RMS_image, binary_image_mat, 32);

		end = clock();

		// Calculating total time taken by the program.
		double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
		cout << "Time taken by program is : " << fixed
			<< time_taken << setprecision(5);
		cout << " sec " << endl;

	
		imshow("RMS image", RMS_image);
		imshow("Feature image", feature_image);
		
		char c = (char)waitKey(10);
		if (c == 27) break; //Press escape to stop program
		
	}

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



	destroyAllWindows();

	return 0;
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

	//cout << "DET: " << to_string(det) << "  HS10: " << to_string(harris_descriptor[1][0]) << "\n";

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