//
// Created by VanHoai on 30/5/25.
//

#include "image_processing.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>

void ImageProcessing::convertToGrayScale() {
    processed = cv::Mat(original.rows, original.cols, CV_8UC1);

    for (int y = 0; y < original.rows; y++) {
        for (int x = 0; x < original.cols; x++) {
            const auto &pixel = original.at<cv::Vec3b>(y, x);

            auto grayVal = static_cast<uchar>(
                0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);

            processed.at<uchar>(y, x) = grayVal;
        }
    }
}
