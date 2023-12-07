

// Stereo matching by rectification
// Martin Schaarup, IA on MC, 11.10.2023
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core.hpp>
using namespace std;
using namespace cv;


Mat readImage(string imgName) {
    // Directory to file - turn \ to /
    string dir = "C:/Users/marti/OneDrive - Danmarks Tekniske Universitet/DTU/Master/30330 Image Analysis MC/"; // /test_pics/stereo_match_check/"; // "C : / Users / marti / Downloads / ";

    // Read image
    Mat image = imread(dir + imgName, IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cerr << "Could not open or find the image." << std::endl;

    }
    normalize(image, image, 0, 255, NORM_MINMAX);
    image.convertTo(image, CV_8U); // 8-bit unassigned

    return image;
}


void rectifyPoints(const cv::Point2d& point_left, const cv::Point2d& point_right,
    const cv::Mat& M_int_left, const cv::Mat& M_int_right,
    const cv::Mat& distortionCoeffs1, const cv::Mat& distortionCoeffs2,
    const cv::Mat& R, const cv::Mat& T,
    cv::Point2d& rectified_point_left_out, cv::Point2d& rectified_point_right_out)
{
    // Specify the image size (adjust with your actual image size)
    cv::Size image_size(1280, 960);

    // Compute rectification transform
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(M_int_left, distortionCoeffs1, M_int_right, distortionCoeffs2, image_size, R, T, R1, R2, P1, P2, Q);

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

/**
* Returns the depth of a given point from the left image by searching in the right image
*
* @param ul input point column position
* @param vl input point column position
* @param grayL input image of the left camera
* @param grayR input image of the right camera
* @return z output depth
*/
void getDepth(int ul, int vl, Mat gray_l, Mat gray_r, double& x, double& y, double& z) { // måske integrer 50 som en parameter


    // Camera intrinsic parameters (adjust these based on your camera setup)
//Mat K1, K2;       // Camera calibration matrices -> camera matrix = [fx 0 0]^T , [0 fy 0]^T , [cx cy 1]^T]

// Camera 1 Intrinsics
    cv::Mat M_int_left = (cv::Mat_<double>(3, 3) <<
        1083.6432, 0, 622.0939,
        0, 1081.5531, 464.5388,
        0, 0, 1
        ); // pixels
    cv::Mat distCoeffs1 = (cv::Mat_<double>(5, 1) << -0.3936, 0.1667, 0, 0, 0);

    // Camera 2 Intrinsics
    cv::Mat M_int_right = (cv::Mat_<double>(3, 3) <<
        1169.0548, 0, 635.6450,
        0, 1165.6900, 467.2909,
        0, 0, 1
        ); // pixels
    cv::Mat distCoeffs2 = (cv::Mat_<double>(5, 1) << -0.4002, 0.1880, 0, 0, 0);

    // Position And Orientation of Camera 2 Relative to Camera 1
    cv::Mat R = (cv::Mat_<double>(3, 3) <<
        0.999953, -0.0094, 0.0033,
        0, 1.0000, 0.0000,
        -0.0033, -0.0094, 0.999953
        );
    cv::Mat T = (cv::Mat_<double>(3, 1) << -249.0464, -0.5551, 0.9119); // mm




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
    int window_size = 50;
    // Scan through a band in the right image on a horizontal line
    int band_height = 30;
    // Define the Mats outside of if-else conditions
    Mat dst_l, window;
    int hori_start, vert_start_w, vert_start_b;
    if (ul < window_size / 2) {
        if (vl < window_size / 2) {
            dst_l = Mat(gray_l, Rect(ul, vl, window_size, window_size));
            hori_start = 0;
            vert_start_w = 0;
            vert_start_b = 0;
        }
        else if (vl > gray_l.rows - 1 - window_size / 2) {
            dst_l = Mat(gray_l, Rect(ul, vl - window_size, window_size, window_size));
            hori_start = 0;
            vert_start_w = window_size;
            vert_start_b = band_height;
        }
        else {
            dst_l = Mat(gray_l, Rect(ul, vl - window_size / 2, window_size, window_size));
            hori_start = 0;
            vert_start_w = window_size / 2;
            vert_start_b = band_height / 2;
        }
    }
    else if (ul > gray_l.cols - 1 - window_size / 2) {
        if (vl < window_size / 2) {
            dst_l = Mat(gray_l, Rect(ul - window_size, vl, window_size, window_size));
            hori_start = window_size;
            vert_start_w = 0;
            vert_start_b = 0;
        }
        else if (vl > gray_l.rows - 1 - window_size / 2) {
            dst_l = Mat(gray_l, Rect(ul - window_size, vl - window_size, window_size, window_size));
            hori_start = window_size;
            vert_start_w = window_size;
            vert_start_b = band_height;
        }
        else {
            dst_l = Mat(gray_l, Rect(ul - window_size, vl - window_size / 2, window_size, window_size));
            hori_start = window_size;
            vert_start_w = window_size / 2;
            vert_start_b = band_height / 2;
        }
    }
    else {
        if (vl < window_size / 2) {
            dst_l = Mat(gray_l, Rect(ul - window_size / 2, vl, window_size, window_size));
            hori_start = window_size / 2;
            vert_start_w = 0;
            vert_start_b = 0;
        }
        else if (vl > gray_l.rows - 1 - window_size / 2) {
            dst_l = Mat(gray_l, Rect(ul - window_size / 2, vl - window_size, window_size, window_size));
            hori_start = window_size / 2;
            vert_start_w = window_size;
            vert_start_b = band_height;
        }
        else {
            dst_l = Mat(gray_l, Rect(ul - window_size / 2, vl - window_size / 2, window_size, window_size));
            hori_start = window_size / 2;
            vert_start_w = window_size / 2;
            vert_start_b = band_height / 2;
        }
    }

    window = dst_l.clone();  // Assign a clone of dst_l to window

    // Variables to store the best matching region
    Rect best_patch;
    int ur, best_m = INT_MAX;
    int vr;
    // Search through band and horizontal line to the left only:
    for (int i = 0; i < band_height; i++) {
        for (int col = hori_start; col <= ul; col++) { // col < gray_r.cols + hori_start - window_size
            Mat dst_r(gray_r, Rect(col - hori_start, vl - vert_start_w - vert_start_b + i, window_size, window_size)); // x, y, width, height
            // Sum of absolute differences (SAD):
            int m = cv::sum(abs(window - dst_r.clone()))[0];
            // Best match in scan gets saved
            // Update the best matching region
            if (col == hori_start || m < best_m) {
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

    rectifyPoints(point_left, point_right, M_int_left, M_int_right, distCoeffs1, distCoeffs2, R, T, point_rect_left, point_rect_right);

    int ul_rec = point_rect_left.x;
    int vl_rec = point_rect_left.y;
    int ur_rec = point_rect_right.x;
    int vr_rec = point_rect_right.y;
    cout << "\n HEJ: " + to_string(ul_rec) + to_string(vl_rec) + to_string(ur_rec) + to_string(vr_rec) + " \n\n\n";

    // Caluclate x, y, z and d (disparity):
    double d = ul_rec - ur_rec; // ul - ur
    double doff = (cxl - cxr) / 2;
    z = b * fxl / (d + doff) / 1000; //mm -> m 
    x = (ul - cxl) * z / fxl;
    y = (vl - cyl) * z / fyl;

    // UNCOMMENT FROM HERE
    // Draw a rectangle around the best matching region
    rectangle(gray_r, best_patch, Scalar(255), 2);
    // Draw a rectangle around the region in the left image that corresponds to the matching region in the right image
    Rect draw_l(ul - hori_start, vl - window_size / 2, best_patch.width, best_patch.height); // NOT ALL ENDPOINTS
    rectangle(gray_l, draw_l, Scalar(255), 2);

    // Display both the left and right images with matching regions highlighted
    // Resize to fit
    float width = 640; //pixel width
    double scale = float(width) / gray_l.size().width;
    Mat disp_l, disp_r;
    resize(gray_l, disp_l, cv::Size(0, 0), scale, scale);
    resize(gray_r, disp_r, cv::Size(0, 0), scale, scale);
    Mat disp_img;
    hconcat(disp_l, disp_r, disp_img);
    cv::imshow("Matching Result - L and R", disp_img);

    cout << "\ndisparity " + std::to_string(d) + "\n";
    cout << " x: " + std::to_string(x) + " y: " + std::to_string(y) + " z: " + std::to_string(z);
    cout << "\nul: " + std::to_string(ul) + " vl: " + std::to_string(vl) + " ur: " + std::to_string(ur) + " vr: " + std::to_string(vr);
    cout << "\nul_rec: " + std::to_string(ul_rec) + " vl_rec: " + std::to_string(vl_rec) + " ur_rec: " + std::to_string(ur_rec) + " vr_rec: " + std::to_string(vr_rec);
    // TO HERE
}

int main()
{
    // Image name, read and show:
    string imgNameL = "calib_pics_l/xl1.png";
    Mat grayL = readImage(imgNameL);
    string imgNameR = "calib_pics_r/xr1.png";
    Mat grayR = readImage(imgNameR);

    // Search through picture with ROI to find the position of best match in the image
    int x_pos = 610; //0 - 1279 // 610
    int y_pos = 550; //0 - 959 // 550
    double x, y, z;
    getDepth(x_pos, y_pos, grayL, grayR, x, y, z);

    waitKey(0);
    return 0;
}
