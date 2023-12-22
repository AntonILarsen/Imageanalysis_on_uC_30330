#include "FeatureDetection.h"

// For the sake of the other guys, we must use an odd X odd kernel size. We shall return the top left coordinate of the kernel plus (kernel size - 1)/2
const int x_bin_n = 45; //Variables for setting the number f x and y bins in which
const int y_bin_n = 34; //we find the RMS value to judge feasibility of feature recognition

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

	cap.set(CAP_PROP_EXPOSURE, -6.0);
	cap.set(CAP_PROP_FRAME_WIDTH, 320);
	cap.set(CAP_PROP_FRAME_HEIGHT, 240);
	
	cout << "Exposure value: " << to_string(cap.get(CAP_PROP_EXPOSURE)) << '\n';
	cout << "Gain value: " << to_string(cap.get(CAP_PROP_GAIN)) << '\n';

	cap >> output;

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

	//Show Camera output
	while (1) {
		cap >> output;

		start = clock();

		uint8_3_bgr_to_8_bit_gray(output, BW_image);
		GaussianBlur(BW_image, image_1, Size(5, 5), 0); //Totally mandatory in the precense of noise

		//Finds good corner features using FAST and puts the coordinates into list_Old
		feature_detection2(image_1, list, 20); 

		//Finds matches of features listed in list and list_Old in image_2 and image_1, respectively,
		//and reorders list_Old, such that matching features have the same index
		feature_ordering(image_2, image_1, list_Old, list, 200, delta_x, delta_y);

		plotTracking(image_1, image_2, list, list_Old); //Plots lines between new and old features

		image_2 = image_1.clone();  //Set image_2 (the old image) equal to image_1, such that we have a succesion of images

		feature_detection2(image_2, list_Old, 20);

		end = clock();

		//https://www.geeksforgeeks.org/how-to-measure-time-taken-by-a-program-in-c/
		double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
		/*cout << "Time taken by program is : " << fixed
			<< time_taken << setprecision(5);
		cout << " sec " << endl;*/

		char c = (char)waitKey(10);
		if (c == 27) break; //Press escape to stop program
	}

	destroyAllWindows();
	return 0;
}

//Finds features in the image and reduces the number based on the best features found. Returns the number of found features
int feature_detection2(Mat image, featurePoint list[NMAX], double threshold)
{
	std::vector<cv::KeyPoint> keypoints_fast;
	cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create((int)threshold);

	fast->detect(image, keypoints_fast);
	int sizeOfList = (int)size(keypoints_fast);

	int minimumScore = INT_MAX;
	int maximumScore = INT_MIN;
	//Finds the min and max of the feature list 'response' value from openCV FAST
	for (int i = 0; i < sizeOfList; i++)
	{
		int responseValue = keypoints_fast[i].response;
		if (responseValue < minimumScore)
			minimumScore = responseValue;

		if (responseValue > maximumScore)
			maximumScore = responseValue;
	}

	//If the maximum and minimum scores are the same, or if minimum is somehow greater than maximum, then end the function
	if (maximumScore - minimumScore < 1)
		return 0;

	const int histResolution = 256;
	int scoreHistogram[histResolution];
	for (int i = 0; i < histResolution; i++)
		scoreHistogram[i] = 0;

	//Creates a histogram of the number of features with a specific score with histResolution number of bins
	for (int i = 0; i < sizeOfList; i++)
	{
		//Normalises the keypoints_fast[i].response -value to fit within 0 and histResolution - 1
		int scaledIndex = (keypoints_fast[i].response - minimumScore) * (histResolution - 1) / (maximumScore - minimumScore);
		scoreHistogram[scaledIndex] += 1; //Use the response value as an index to create a histogram
	}

	int sum = 0;
	int thresholdForLessThanNMAXFeatures = 0;
	//Creates a cumulative probability distrubution and returns the FAST threshold value for which
	//there are just less than NMAX features
	for (int sumIndex = 0; sumIndex < histResolution; sumIndex++)
	{
		sum += scoreHistogram[sumIndex];

		if (sum >= NMAX)
		{
			thresholdForLessThanNMAXFeatures = ((sumIndex - 1) * (maximumScore - minimumScore) + minimumScore * histResolution - minimumScore) / histResolution;
			break;
		}
	}

	if (sum < NMAX) //If we find less than NMAX features, we just use all the FAST features
		thresholdForLessThanNMAXFeatures = INT_MAX;

	int indexForList = 0;
	for (int i = 0; i < sizeOfList; i++)
	{
		int responseValue = keypoints_fast[i].response;

		//If one of the keypoints have a lower (better) score than the score limit, save it in the list
		//The cumulative sum approach from before makes sure that indexForList never goes beyond NMAX,
		//so checking of indexForlist < NMAX isn't necessary
		if (responseValue < thresholdForLessThanNMAXFeatures)
		{
			list[indexForList].x = (int)keypoints_fast[i].pt.x;
			list[indexForList].y = (int)keypoints_fast[i].pt.y;
			list[indexForList].active = true;
			indexForList++;
		}
	}
	//Sets the remaning list to inactive to not have points 'hanging'
	if (indexForList < NMAX)
	{
		for (int i = indexForList; i < NMAX; i++)
			list[i].active = false;
	}

	return indexForList;
}

int feature_ordering(Mat image_old, Mat image_latest, featurePoint list_old[NMAX], featurePoint list_latest[NMAX], int threshold, int delta_x, int delta_y)
{
	featurePoint list_buffer[NMAX];
	int old_Score;
	int score = 0;
	int half_delta_x = (int)(delta_x / 2);
	int half_delta_y = (int)(delta_y / 2);
	int x_res = image_latest.cols;
	int y_res = image_latest.rows;
	int num_matches = 0;

	//Iterates through the new list of FAST features
	for (int ne = 0; ne < NMAX; ne++) 
	{
		int bestMatchIndex = 0;
		old_Score = INT_MAX;

		//Clearing the buffer list to prevent false positive actives from lack of initialisation
		list_buffer[ne].active = false; 

		//Iterates through the old list of FAST features
		for (int ol = 0; ol < NMAX; ol++) 
		{
			if (list_old[ol].active) //Checking to see if the feature is actually active
			{
				int position_x = (int)list_latest[ne].x;
				int position_y = (int)list_latest[ne].y;
				int position_x_old = (int)list_old[ol].x;
				int position_y_old = (int)list_old[ol].y;

				if ((position_x - delta_x > 0) && (position_x + delta_x < x_res) //If we can fit the image kernel within image borders
					&& (position_y - delta_y > 0) && (position_y + delta_y < y_res) //Should be half kernel, actually
					&& (position_x_old - delta_x > 0) && (position_x_old + delta_x < x_res)
					&& (position_y_old - delta_y > 0) && (position_y_old + delta_y < y_res))
				{
					score = 0;

					//Compute the sum of absolute difference between the selected features/image patches
					for (int x = 0; x < delta_x; x++)
					{
						for (int y = 0; y < delta_y; y++)
						{
							score += (int)abs(image_old.at<uchar>(Point(position_x_old + x - half_delta_x, position_y_old + y - half_delta_x))
								- image_latest.at<uchar>(Point(position_x + x - half_delta_x, position_y + y - half_delta_x)));
						}
					}

					if (score < old_Score) //Save score and index if better than previous
					{
						old_Score = score;
						bestMatchIndex = ol;
					}
				}
			}
		}

		if (old_Score < threshold) //If the score isn't very bad, it can be used
		{
			list_buffer[ne].x = list_old[bestMatchIndex].x; //Save the coordinates of the feature from the old
			list_buffer[ne].y = list_old[bestMatchIndex].y; //list which best matches the new feature
			list_buffer[ne].score = old_Score; //Save the score
			list_buffer[ne].active = true; //Enable the feature point

			list_old[bestMatchIndex].active = false; //Disable the found feature from the old list to not use it again

			num_matches += 1;
		}
		else
		{
			list_buffer[ne].active = false;
		}
	}

	for (int i = 0; i < NMAX; i++)
	{
		list_old[i] = list_buffer[i]; //Copy the buffered list into the original
		list_latest[i].active = list_buffer[i].active; //Make sure only features which were matched are active in each list
		list_latest[i].score = list_buffer[i].score; //Copy the score (as it is a common value between old and new feature)
	}

	return num_matches;
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