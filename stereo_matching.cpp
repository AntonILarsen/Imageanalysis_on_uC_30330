// Stereo matching by rectification
// Martin Schaarup, IA on MC, 11.10.2023
#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
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

/**
* Returns the depth of a given point from the left image by searching in the right image
*
* @param ul input point column position
* @param vl input point column position
* @param grayL input image of the left camera
* @param grayR input image of the right camera
* @param x output position x-coordinate
* @param y output position y-coordinate
* @param z output position z-coordinate
*/
void getDepth(int ul, int vl, Mat gray_l, Mat gray_r, double& x, double& y, double& z) { // måske integrer 50 som en parameter

    // Constants for camera parameters
    // baseline b:
    double b = 249.0464; // mm
    // focal length (fx,fy):
    double fxl = 1083.6432; // pixels
    double fyl = 1081.5531;
    // optical centers (cx,cy):
    double cxl = 622.0939; // pixels
    double cyl = 464.5388;
    double cxr = 635.6450;
    double cyr = 467.2909;

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
    int ur, vr, best_m = INT_MAX;
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

    // Caluclate x, y, z and d (disparity):
    double d = ul - ur; // ul - ur
    double doff = (cxl - cxr) / 2;
    z = b * fxl / (d + doff);
    x = (ul - cxl) * z / fxl;
    y = (vl - cyl) * z / fyl;


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
    cout << " x: " + std::to_string(x / 1000) + " y: " + std::to_string(y / 1000) + " z: " + std::to_string(z / 1000);
    cout << "\nul: " + std::to_string(ul) + " vl: " + std::to_string(vl) + " ur: " + std::to_string(ur) + " vr: " + std::to_string(vr);

}

void rectifyStereo(cv::InputArray grayL, cv::InputArray grayR, cv::OutputArray rectL, cv::OutputArray rectR) {
    // Camera intrinsic parameters (adjust these based on your camera setup)
    //Mat K1, K2;       // Camera calibration matrices -> camera matrix = [fx 0 0]^T , [0 fy 0]^T , [cx cy 1]^T]

    // Camera 1 Intrinsics
    cv::Mat cameraMatrix1 = (cv::Mat_<double>(3, 3) <<
        1083.6432, 0, 622.0939,
        0, 1081.5531, 464.5388,
        0, 0, 1
        ); // pixels
    cv::Mat distCoeffs1 = (cv::Mat_<double>(5, 1) << -0.3936, 0.1667, 0, 0, 0);

    // Camera 2 Intrinsics
    cv::Mat cameraMatrix2 = (cv::Mat_<double>(3, 3) <<
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

int main()
{
    // Image name, read and show:
    string imgNameL = "calib_pics_l/xl1.png";
    Mat grayL = readImage(imgNameL);
    string imgNameR = "calib_pics_r/xr1.png";
    Mat grayR = readImage(imgNameR);

    Mat rectL, rectR;
    rectifyStereo(grayL, grayR, rectL, rectR);

    // Search through picture with ROI to find the position of best match in the image
    int ul = 590; //0 - 1279 // 610
    int vl = 548; //0 - 959 // 550
    double x, y, z;
    getDepth(ul, vl, rectL, rectR, x, y, z);

    waitKey(0);
    return 0;
}




