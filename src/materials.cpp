#include "materials.hpp"

#include <CoreGraphics/CGDisplayConfiguration.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace materials {
    // Function to get the screen resolution
    std::pair<int, int> getScreenResolution() {
        auto mainDisplayId = CGMainDisplayID();
        int width = CGDisplayPixelsWide(mainDisplayId);
        int height = CGDisplayPixelsHigh(mainDisplayId);
        return std::make_pair(width, height);
    }

    std::pair<int, int> getCenterPosition(int windowWidth, int windowHeight) {
        auto [screenWidth, screenHeight] = getScreenResolution();
        int x = (screenWidth - windowWidth) / 2;
        int y = (screenHeight - windowHeight) / 2;
        return std::make_pair(x, y);
    }

    /**
     * Converts a color image to grayscale using the standard formula.
     * The formula is: gray = 0.299 * R + 0.587 * G + 0.114 * B
     */
    Mat convertToGrayscale(const Mat &input) {
        Mat gray(input.rows, input.cols, CV_8UC1);

        for (int y = 0; y < input.rows; y++) {
            for (int x = 0; x < input.cols; x++) {
                Vec3b pixel = input.at<Vec3b>(y, x);
                uchar grayVal = static_cast<uchar>(
                    0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);

                gray.at<uchar>(y, x) = grayVal;
            }
        }

        return gray;
    }

    void showPairImage(int widthSingle, int heightSingle,
                       pair<Mat, Mat> sources) {
        cv::namedWindow("CenteredWindow");

        int width = 2 * widthSingle;
        int height = 1 * heightSingle;
        cv::Mat images = cv::Mat(height, width, sources.first.type());

        auto [x, y] = materials::getCenterPosition(width, height);

        cv::Rect subImageROI = cv::Rect(0, 0, heightSingle, widthSingle);
        sources.first.copyTo(images(subImageROI));

        subImageROI.x = heightSingle;
        sources.second.copyTo(images(subImageROI));

        cv::imshow("CenteredWindow", images);
        cv::moveWindow("CenteredWindow", x, y - 100);
        cv::waitKey(0);
    }

}   // namespace materials
