//
// File        : main.cpp
// Author      : Hinsun
// Date        : 2025-06-25
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "core/common.hpp"
#include "core/core.hpp"
#include "core/tensor.hpp"
#include "processing/features.hpp"

std::string path = "/Users/hinsun/Workspace/ComputerScience/VisionCpp/assets/workspace.png";

int main() {
    const cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "Cannot read image from " << path << std::endl;
        return EXIT_FAILURE;
    }

    core::Tensor<core::float32> tensor;
    core::matToTensor(image, tensor);

    processing::SIFT detector;
    detector.detectAndCompute(tensor);

    return EXIT_SUCCESS;
}
