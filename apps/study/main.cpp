//
// File        : main.cpp
// Author      : Hinsun
// Date        : 2025-06-25
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#include <iostream>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "core/common.hpp"
#include "core/core.hpp"
#include "core/tensor.hpp"
#include "processing/features.hpp"

std::string path = "/Users/hinsun/Workspace/ComputerScience/VisionCpp/assets/workspace.png";

std::vector<cv::KeyPoint> convertToKeypointsCV(const std::vector<processing::Keypoint> &keypoints,
                                               const cv::Size &imageSize) {
    std::vector<cv::KeyPoint> kps;

    for (const auto &kp : keypoints) {
        cv::KeyPoint keypoint;
        keypoint.pt.x = kp.x;
        keypoint.pt.y = kp.y;
        keypoint.size = kp.scale * 2.0f;
        keypoint.angle = kp.angle * 180.0f / CV_PI;
        if (keypoint.angle < 0)
            keypoint.angle += 360.0f;

        keypoint.octave = (kp.octave << 8) | (kp.layer & 0xFF);
        keypoint.response = 1.0f;
        keypoint.class_id = 0;

        if (keypoint.pt.x < 0 || keypoint.pt.x >= imageSize.width || keypoint.pt.y < 0 ||
            keypoint.pt.y >= imageSize.height) {
            continue;
        }
        kps.push_back(keypoint);
    }

    return kps;
}

void detectWithOpenCV() {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "Cannot read image from " << path << std::endl;
        return;
    }

    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
    std::cout << "OpenCV SIFT found: " << keypoints.size() << " keypoints." << std::endl;
}

int main() {
    const cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "Cannot read image from " << path << std::endl;
        return EXIT_FAILURE;
    }

    core::Tensor<core::float32> tensor;
    core::matToTensor(image, tensor);

    const std::vector<processing::Keypoint> keypoints = processing::SIFT::detectAndCompute(tensor);
    std::cout << "SIFT: " << keypoints.size() << " keypoints detected" << std::endl;

    const std::vector<cv::KeyPoint> cvKeypoints = convertToKeypointsCV(keypoints, image.size());
    std::cout << "Converted to OpenCV keypoints: " << cvKeypoints.size() << std::endl;

    cv::Mat output;
    cv::drawKeypoints(image, cvKeypoints, output, cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    core::showImageCenterWindow(output, "Keypoints Detected by SIFT");
    return EXIT_SUCCESS;
}
