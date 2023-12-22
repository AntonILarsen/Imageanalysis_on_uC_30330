

// Stereo matching by rectification
// Martin Schaarup, IA on MC, 11.10.2023
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core.hpp>
#include <string>
using namespace std;
using namespace cv;

#include <sstream>
#include <iomanip>
const float UNI_SCALE = 0.25;


std::string to_string_with_precision(double value, int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

Mat readImage(string imgName) {
    // Directory to file - turn \ to /
    string dir = "C:/Users/marti/OneDrive - Danmarks Tekniske Universitet/DTU/Master/30330 Image Analysis MC/"; // /test_pics/stereo_match_check/"; // "C : / Users / marti / Downloads / ";

    // Read image
    Mat image = imread(dir + imgName, IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Could not open or find the image." << std::endl;

    }
   // normalize(image, image, 0, 255, NORM_MINMAX);
    image.convertTo(image, CV_8U); // 8-bit unassigned

    return image;
}

void rectifyPoints(const cv::Point2d& point_left, const cv::Point2d& point_right,
    const cv::Mat& M_int_left, const cv::Mat& M_int_right,
    const cv::Mat& distortionCoeffs1, const cv::Mat& distortionCoeffs2,
    const cv::Mat& R, const cv::Mat& T, cv::Mat& R1, cv::Mat& R2, cv::Mat& P1, cv::Mat& P2,
    cv::Point2d& rectified_point_left_out, cv::Point2d& rectified_point_right_out)
{
    // Undistort and rectify the points
    std::vector<cv::Point2d> points_left = { point_left };
    std::vector<cv::Point2d> points_right = { point_right };

    cv::Mat rectified_points_left, rectified_points_right;
    cv::undistortPoints(points_left, rectified_points_left, M_int_left, distortionCoeffs1, R1, P1);
    cv::undistortPoints(points_right, rectified_points_right, M_int_right, distortionCoeffs2, R2, P2);

    // Extract rectified points
    cv::Point2d rectified_point_left = rectified_points_left.at<cv::Point2d>(0);
    cv::Point2d rectified_point_right = rectified_points_right.at<cv::Point2d>(0);

    // Set the rectified points
    rectified_point_left_out = rectified_point_left;
    rectified_point_right_out = rectified_point_right;
    // Now point_rect_left_tuple and point_rect_right_tuple contain the rectified coordinates for the specified points
}


void loadParams(cv::Mat& M_int_left, cv::Mat& M_int_right,
    cv::Mat& distortionCoeffs1, cv::Mat& distortionCoeffs2,
    cv::Mat& R, cv::Mat& T, cv::Mat& R1, cv::Mat& R2, cv::Mat& P1, cv::Mat& P2) {

    // Camera intrinsic parameters (adjust these based on your camera setup)
//Mat K1, K2;       // Camera calibration matrices -> camera matrix = [fx 0 0]^T , [0 fy 0]^T , [cx cy 1]^T]

    // Camera 1 Intrinsics
    M_int_left = (cv::Mat_<double>(3, 3) <<
        1043.7984551501 * UNI_SCALE, 0, 642.556283726675 * UNI_SCALE,
        0, 1040.95336826827 * UNI_SCALE, 460.665677233496 * UNI_SCALE,
        0, 0, 1
        ); // pixels
    distortionCoeffs1 = (cv::Mat_<double>(5, 1) << -0.378359683169019, 0.206282045446043, 0, 0, 0);

    // Camera 2 Intrinsics
    M_int_right = (cv::Mat_<double>(3, 3) <<
        1117.08308347549 * UNI_SCALE, 0, 634.601065459339 * UNI_SCALE,
        0, 1114.05404906178 * UNI_SCALE, 468.71949855429 * UNI_SCALE,
        0, 0, 1
        ); // pixels
    distortionCoeffs2 = (cv::Mat_<double>(5, 1) << -0.374917895766859, 0.183645216412797, 0, 0, 0);

    // Position And Orientation of Camera 2 Relative to Camera 1
    R = (cv::Mat_<double>(3, 3) <<
        0.99996031435586, -0.00486793544137724, -0.00746142867469252,
        0.00486519436667386, 0.99998809057628, -0.000385472929552811,
        0.00746321627071234, 0.000349156331055901, 0.999972088856861
        );
    T = (cv::Mat_<double>(3, 1) << -249.035396994937, -0.539458297110969, 1.24768572279461); // mm

    // Specify the image size (adjust with your actual image size)
    cv::Size image_size(1280 * UNI_SCALE, 960 * UNI_SCALE);

    // Compute rectification transform
    Mat Q;
    cv::stereoRectify(M_int_left, distortionCoeffs1, M_int_right, distortionCoeffs2, image_size, R, T, R1, R2, P1, P2, Q);
 
}

/**
* Returns the depth of a given point from the left image by searching in the right image
*
* @param ul input point column position
* @param vl input point column position
* @param grayL input image of the left camera
* @param grayR input image of the right camera
* @return z output depth
*/

void getDepth(int ul, int vl, Mat gray_l, Mat gray_r,Mat disp_l, Mat disp_r, Mat M_int_left, Mat M_int_right, Mat distCoeffs1, Mat distCoeffs2,
    Mat R, Mat T, Mat R1, Mat R2, Mat P1, Mat P2, double& x, double& y, double& z, int plotStereoMatching) { // måske integrer 50 som en parameter

    // Constants for camera parameters
    // baseline b:
    double b = -T.at<double>(0); // mm
    // focal length (fx,fy):
    double fxl = M_int_left.at<double>(0, 0); // pixels
    double fyl = M_int_left.at<double>(1, 1);
    // optical centers (cx,cy):
    double cxl = M_int_left.at<double>(0, 2); // pixels
    double cyl = M_int_left.at<double>(1, 2);
    double cxr = M_int_right.at<double>(0, 2);
    double cyr = M_int_right.at<double>(1, 2);

    // Handling boundary cases
    int window_size = (int)round(64.0 * UNI_SCALE);
    // Scan through a band in the right image on a horizontal line
    int band_height = (int)round(32.0 * UNI_SCALE);
    // Define the Mats outside of if-else conditions
    Rect dst_l;
    Mat window;
    int hori_start, vert_start_w, vert_start_b;
    if (ul < window_size / 2) {
        if (vl < window_size / 2) {
            dst_l = Rect(ul, vl, window_size, window_size);
            hori_start = 0;
            vert_start_w = 0;
            vert_start_b = 0;
        }
        else if (vl > gray_l.rows - 1 - window_size / 2) {
            dst_l = Rect(ul, vl - window_size, window_size, window_size);
            hori_start = 0;
            vert_start_w = window_size;
            vert_start_b = band_height;
        }
        else {
            dst_l = Rect(ul, vl - window_size / 2, window_size, window_size);
            hori_start = 0;
            vert_start_w = window_size / 2;
            vert_start_b = band_height / 2;
        }
    }
    else if (ul > gray_l.cols - 1 - window_size / 2) {
        if (vl < window_size / 2) {
            dst_l = Rect(ul - window_size, vl, window_size, window_size);
            hori_start = window_size;
            vert_start_w = 0;
            vert_start_b = 0;
        }
        else if (vl > gray_l.rows - 1 - window_size / 2) {
            dst_l = Rect(ul - window_size, vl - window_size, window_size, window_size);
            hori_start = window_size;
            vert_start_w = window_size;
            vert_start_b = band_height;
        }
        else {
            dst_l = Rect(ul - window_size, vl - window_size / 2, window_size, window_size);
            hori_start = window_size;
            vert_start_w = window_size / 2;
            vert_start_b = band_height / 2;
        }
    }
    else {
        if (vl < window_size / 2) {
            dst_l = Rect(ul - window_size / 2, vl, window_size, window_size);
            hori_start = window_size / 2;
            vert_start_w = 0;
            vert_start_b = 0;
        }
        else if (vl > gray_l.rows - 1 - window_size / 2) {
            dst_l = Rect(ul - window_size / 2, vl - window_size, window_size, window_size);
            hori_start = window_size / 2;
            vert_start_w = window_size;
            vert_start_b = band_height;
        }
        else {
            int left = max(ul - window_size / 2, 0); // Ensure left edge is not negative
            int top = max(vl - window_size / 2, 0);  // Ensure top edge is not negative
            int right = min(left + window_size, gray_l.cols - 1); // Ensure right edge does not go beyond image width
            int bottom = min(top + window_size, gray_l.rows - 1); // Ensure bottom edge does not go beyond image height

            int adjustedWidth = right - left;
            int adjustedHeight = bottom - top;
            
            dst_l = Rect(left, top, adjustedWidth, adjustedHeight);

            // Calculate the starting horizontal and vertical position
            if (ul == 295 && vl == 225)
                cout << endl;
            hori_start = ul - left;
            vert_start_w = vl - top;
            vert_start_b = (band_height - adjustedHeight) / 2; // Adjust band height if necessary
        }
    } 

    Mat sample = Mat(gray_l, dst_l);
    window = sample.clone();  // Assign a clone of dst_l to window

    // Variables to store the best matching region
    Rect best_patch;
    int ur, best_m = INT_MAX;
    int vr, m;
    // Search through band and horizontal line to the left only:
    for (int i = 0; i < band_height; i++) {
        for (int col = hori_start; col <= ul; col++) { // col < gray_r.cols + hori_start - window_size
            // Sum of absolute differences (SAD):
            int correctionStuff = 0;
            if (vl - vert_start_w - vert_start_b + i + window_size > gray_l.rows - 1)
            {
                correctionStuff = gray_l.rows - (vl - vert_start_w - vert_start_b + i + window_size);
            }
                m = cv::sum(abs(window - gray_r(Rect(col - hori_start, vl - vert_start_w - vert_start_b + i + correctionStuff, window_size, window_size))))[0]; // x, y, width, height
            

            // Best match in scan gets saved
            // Update the best matching region
            if (/*col == hori_start ||*/ m < best_m) {
                ur = col;
                vr = vl + i - vert_start_b;
                best_m = m;
                best_patch = Rect(col - hori_start, vl - vert_start_w - vert_start_b + i, window_size, window_size);
            }
        }
    }

    // Points for undistotion and rectification:
    cv::Point2d point_left(ul, vl);  // Coordinates in the left image
    cv::Point2d point_right(ur, vr);  // Coordinates in the right imageur: 227 vr: 674
    cv::Point2d point_rect_left;
    cv::Point2d point_rect_right;

    rectifyPoints(point_left, point_right, M_int_left, M_int_right, distCoeffs1, distCoeffs2, R, T, R1, R2, P1, P2, point_rect_left, point_rect_right);
    
    int ul_rec = point_rect_left.x;
    int vl_rec = point_rect_left.y;
    int ur_rec = point_rect_right.x;
    int vr_rec = point_rect_right.y;
    //cout << "\n HEJ: " + to_string(ul_rec) + to_string(vl_rec) + to_string(ur_rec) + to_string(vr_rec) + " \n\n\n";

    // Caluclate x, y, z and d (disparity):
    double d = ul_rec - ur_rec; // ul - ur
    double doff = (cxl - cxr) / 2;
    z = b * fxl / (d + doff) / 1000; //mm -> m 
    x = (ul_rec - cxl) * z / fxl;
    y = (vl_rec - cyl) * z / fyl;

    // UNCOMMENT FROM HERE
    if(plotStereoMatching){
    // Draw a rectangle around the best matching region
    // Generate random color
    Scalar color(rand() % 256, rand() % 256, rand() % 256);

    // Display both the left and right images with matching regions highlighted
    // Resize to fit
    //float width = 640; //pixel width
    //double scale = float(width) / gray_l.size().width;
    //Mat disp_l, disp_r;
    //resize(gray_l, disp_l, cv::Size(), scale, scale);
    //resize(gray_r, disp_r, cv::Size(), scale, scale);

    // Draw a rectangle with random color
  //  best_patch.x = (best_patch.x + best_patch.width / 2) * scale - best_patch.width / 2;
  //  best_patch.y = (best_patch.y + best_patch.height / 2) * scale - best_patch.height / 2;

    cv::rectangle(disp_r, best_patch, color, 1); // -1 indicates filled rectangle

    //rectangle(gray_r, best_patch, Scalar(255), 2);
    // Draw a rectangle around the region in the left image that corresponds to the matching region in the right image
    //Rect draw_r((ur)-best_patch.width/2, (vr)-best_patch.height/2, best_patch.width, best_patch.height); // NOT ALL ENDPOINTS
    cv::rectangle(disp_l, dst_l, color, 1);
    
    cv::putText(disp_l, //target image
        to_string_with_precision(z, 1), //text
        cv::Point(dst_l.x, -3 + dst_l.y ), //top-left position
        cv::FONT_HERSHEY_SIMPLEX,
        0.4,
        color, //font color
        1.5);

    Mat disp_img;
    cv::hconcat(disp_l, disp_r, disp_img);
    cv::imshow("Matching Result - L and R", disp_img);

    //cout << "\ndisparity " + std::to_string(d) + "\n";
    //cout << " x: " + std::to_string(x) + " y: " + std::to_string(y) + " z: " + std::to_string(z);
    //cout << "\nul: " + std::to_string(ul) + " vl: " + std::to_string(vl) + " ur: " + std::to_string(ur) + " vr: " + std::to_string(vr);
    //cout << "\nul_rec: " + std::to_string(ul_rec) + " vl_rec: " + std::to_string(vl_rec) + " ur_rec: " + std::to_string(ur_rec) + " vr_rec: " + std::to_string(vr_rec);
    }
    // TO HERE
}