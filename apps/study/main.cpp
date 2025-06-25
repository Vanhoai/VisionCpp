//
// File        : main.cpp
// Author      : Hinsun
// Date        : 2025-06-25
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#include <iostream>
#include <opencv2/opencv.hpp>

#include "core/common.hpp"
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
    core::Tensor<core::float32> source;
    core::matToTensor(image, source, 3);
    core::showImageCenterWindow(source, "Source");

    core::Tensor<core::float32> changed;
    processing::Transformations::pad(source, changed, 10, 0.0f);
    core::showImageCenterWindow(changed, "Padded");

    processing::Transformations::crop(source, changed, core::Rect(0, 0, 400, 400));
    core::showImageCenterWindow(changed, "Cropped");

    processing::Transformations::randomCrop(source, changed, 400, 400);
    core::showImageCenterWindow(changed, "Random Cropped");

    processing::Transformations::rotate(source, changed,
                                        processing::Transformations::RotateAngle::CLOCKWISE_90);
    core::showImageCenterWindow(changed, "Rotated 90 degrees");

    processing::Transformations::rotate(source, changed,
                                        processing::Transformations::RotateAngle::CLOCKWISE_180);
    core::showImageCenterWindow(changed, "Rotated 180 degrees");

    processing::Transformations::flip(source, changed, processing::Transformations::FlipCode::BOTH);
    core::showImageCenterWindow(changed, "Flipped Both");

    processing::Transformations::resize(source, changed, 400, 400);
    core::showImageCenterWindow(changed);

    cv::destroyAllWindows();
    return EXIT_SUCCESS;
}
