//
// Created by Hinsun on 2025-06-18
// Copyright (c) 2025 VanHoai. All rights reserved.
//

#include "preprocessing/preprocessing.hpp"

#include <opencv2/opencv.hpp>
#include <stdexcept>

namespace preprocessing {

    void Preprocessor::convertToGrayScale(const cv::Mat& inputImage, cv::Mat& outputImage) {
        // inputImage: BGR format like OpenCV saved
        if (inputImage.empty())
            throw std::invalid_argument("Input image is not empty");

        if (inputImage.channels() != 3)
            throw std::invalid_argument("Input image must be in BGR format (3 channels)");

        // Convert to grayscale using the formula
        // F(R, G, B) = 0.299 * R + 0.587 * G + 0.114 * B

        outputImage = cv::Mat(inputImage.size(), CV_8UC1);
        for (int i = 0; i < inputImage.rows; ++i) {
            for (int j = 0; j < inputImage.cols; ++j) {
                const auto& pixel = inputImage.at<cv::Vec3b>(i, j);
                auto grayValue = static_cast<uchar>(0.299 * pixel[2] +   // R
                                                    0.587 * pixel[1] +   // G
                                                    0.114 * pixel[0]     // B
                );

                outputImage.at<uchar>(i, j) = grayValue;
            }
        }
    }

}   // namespace preprocessing
