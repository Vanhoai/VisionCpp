//
// Created by Hinsun on 2025-06-19.
// Copyright (c) 2025 VanHoai. All rights reserved.
//

#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "processing/transformations.hpp"

std::string path = "/Users/hinsun/Workspace/ComputerScience/VisionCpp/assets/workspace.jpg";

int main() {
    const cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "Cannot read image from " << path << std::endl;
        return EXIT_FAILURE;
    }

    // 736 x 920
    std::cout << "Image size: " << image.size() << std::endl;

    cv::Mat changed;
    processing::Transformations::resize(image, changed, cv::Size(500, 500 * 736 / 920));
    cv::imshow("Workspace", changed);
    cv::waitKey();

    processing::Transformations::normalize(image, changed, cv::Scalar(0, 0, 0),
                                           cv::Scalar(1, 1, 1));
    cv::imshow("Normalized", changed);
    cv::waitKey();

    processing::Transformations::pad(image, changed, cv::Size(20, 20), cv::Scalar(0, 0, 0));
    cv::imshow("Padded", changed);
    cv::waitKey();

    processing::Transformations::crop(image, changed, cv::Rect(100, 100, 300, 300));
    cv::imshow("Cropped", changed);
    cv::waitKey();

    processing::Transformations::randomCrop(image, changed, cv::Size(300, 300));
    cv::imshow("Random Cropped", changed);
    cv::waitKey();

    processing::Transformations::rotate(image, changed,
                                        processing::Transformations::RotateAngle::CLOCKWISE_90);
    cv::imshow("Rotated 90 degrees", changed);
    cv::waitKey();

    processing::Transformations::rotate(image, changed,
                                        processing::Transformations::RotateAngle::CLOCKWISE_180);
    cv::imshow("Rotated 180 degrees", changed);
    cv::waitKey();

    processing::Transformations::flip(image, changed, processing::Transformations::FlipCode::BOTH);
    cv::imshow("Flipped", changed);
    cv::waitKey();

    cv::destroyAllWindows();
    return EXIT_SUCCESS;
}
