#include "materials.hpp"

#include <CoreGraphics/CGDisplayConfiguration.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace materials {

    // Function to get the screen resolution
    std::pair<int, int> getScreenResolution() {
        const auto mainDisplayId = CGMainDisplayID();
        size_t width = CGDisplayPixelsWide(mainDisplayId);
        size_t height = CGDisplayPixelsHigh(mainDisplayId);
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
                const auto &pixel = input.at<Vec3b>(y, x);
                auto grayVal = static_cast<uchar>(
                    0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);

                gray.at<uchar>(y, x) = grayVal;
            }
        }

        return gray;
    }

    void showImageCenterWindow(const Mat &image) {
        const string windowName = "ImageCentered";
        namedWindow(windowName);

        const int width = image.cols;
        const int height = image.rows;

        auto [x, y] = getCenterPosition(width, height);
        imshow(windowName, image);
        moveWindow(windowName, x, y - 100);
        waitKey(0);
        destroyAllWindows();
    }

    void showPairImage(const int widthSingle, const int heightSingle,
                       const pair<Mat, Mat> &sources) {
        const string windowName = "ImageCentered";
        namedWindow(windowName);

        const int width = 2 * widthSingle;
        const int height = 1 * heightSingle;
        const auto images = Mat(height, width, sources.first.type());

        auto [x, y] = getCenterPosition(width, height);

        auto subImageROI = cv::Rect(0, 0, heightSingle, widthSingle);
        sources.first.copyTo(images(subImageROI));

        subImageROI.x = heightSingle;
        sources.second.copyTo(images(subImageROI));

        imshow(windowName, images);
        moveWindow(windowName, x, y - 100);
        waitKey(0);
        destroyAllWindows();
    }

}   // namespace materials
