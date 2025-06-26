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

std::string path = "/Users/hinsun/Workspace/ComputerScience/VisionCpp/assets/workspace.jpg";

int main() {
    const cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "Cannot read image from " << path << std::endl;
        return EXIT_FAILURE;
    }

    // Convert to my tensor
    core::Tensor<core::float32> source;
    core::matToTensor(image, source);

    return EXIT_SUCCESS;
}
