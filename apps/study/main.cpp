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
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "core/common.hpp"
#include "core/core.hpp"
#include "core/tensor.hpp"
#include "processing/features.hpp"

std::string path = "/Users/hinsun/Workspace/ComputerScience/VisionCpp/assets/workspace.png";

void harrisConnerDetection() {
    std::string dream = "/Users/hinsun/Workspace/ComputerScience/VisionCpp/assets/dream.jpg";
    cv::Mat image = cv::imread(dream, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "Please provide an image in this path: " << dream << std::endl;
        return;
    }

    cv::Mat grayscale;
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);

    cv::Mat cornered;
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;

    cv::cornerHarris(grayscale, cornered, blockSize, apertureSize, k);

    cv::Mat output;
    cv::dilate(cornered, output, cv::Mat());

    double threshold = 0.01 * cv::norm(output, cv::NORM_INF);
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            if (output.at<float>(y, x) > threshold) {
                cv::circle(image, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
            }
        }
    }

    core::showImageCenterWindow(image, "Harris Corner Detection");
}

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
    const cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
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

void detectWithCustomSIFT() {
    const cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "Cannot read image from " << path << std::endl;
        return;
    }

    core::Tensor<core::float32> tensor;
    core::matToTensor(image, tensor);

    const std::vector<processing::Keypoint> keypoints = processing::SIFT::detectAndCompute(tensor);
    std::cout << "Custom SIFT found: " << keypoints.size() << " keypoints." << std::endl;

    const std::vector<cv::KeyPoint> cvKeypoints = convertToKeypointsCV(keypoints, image.size());
    std::cout << "Converted to OpenCV keypoints: " << cvKeypoints.size() << std::endl;

    cv::Mat output;
    cv::drawKeypoints(image, cvKeypoints, output, cv::Scalar(0, 255, 0),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    core::showImageCenterWindow(output, "Custom SIFT Keypoints");
}

int main() {
    // const cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    // if (image.empty()) {
    //     std::cout << "Cannot read image from " << path << std::endl;
    //     return EXIT_FAILURE;
    // }

    // core::TensorF32 tensor = core::convertMatToTensor(image);

    // const std::vector<processing::Keypoint> keypoints =
    // processing::ORB::detectAndCompute(tensor); std::cout << "ORB found: " << keypoints.size() <<
    // " keypoints." << std::endl;

    harrisConnerDetection();
    return EXIT_SUCCESS;
}
