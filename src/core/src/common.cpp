//
// File        : common.cpp
// Author      : Hinsun
// Date        : 2025-06-19
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#include "core/common.hpp"

#include <CoreGraphics/CGDisplayConfiguration.h>

#include <opencv2/opencv.hpp>

#include "core/core.hpp"
#include "core/tensor.hpp"

namespace core {

    std::pair<size_t, size_t> getScreenResolution() {
        const auto mainDisplayId = CGMainDisplayID();
        size_t width = CGDisplayPixelsWide(mainDisplayId);
        size_t height = CGDisplayPixelsHigh(mainDisplayId);
        return std::make_pair(width, height);
    }

    void showImageCenterWindow(const cv::Mat &image, const std::string &windowName) {
        cv::namedWindow(windowName);

        const size_t width = image.cols;
        const size_t height = image.rows;

        auto [screenWidth, screenHeight] = getScreenResolution();

        const size_t x = (screenWidth - width) / 2;
        const size_t y = (screenHeight - height) / 2;

        imshow(windowName, image);
        cv::moveWindow(windowName, x, y);

        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    void showImageCenterWindow(const Tensor<float32> &tensor, const std::string &windowName) {
        cv::Mat image;
        tensorToMat(tensor, image);
        showImageCenterWindow(image, windowName);
    }

    void matToTensor(const cv::Mat &src, Tensor<float32> &tensor) {
        if (src.empty())
            throw std::runtime_error("Image should be non-empty");

        if (src.channels() == 1)
            tensor = Tensor<float32>(src.rows, src.cols);
        else
            tensor = Tensor<float32>(src.rows, src.cols, 3);

        for (int y = 0; y < src.rows; ++y) {
            for (int x = 0; x < src.cols; ++x) {
                if (src.channels() == 1) {
                    // Grayscale
                    tensor.at(y, x) = static_cast<float32>(src.at<uchar>(y, x));
                } else if (src.channels() == 3) {
                    // Color
                    const auto &pixel = src.at<cv::Vec3b>(y, x);

                    tensor.at(y, x, 0) = static_cast<float32>(pixel[0]);   // Blue
                    tensor.at(y, x, 1) = static_cast<float32>(pixel[1]);   // Green
                    tensor.at(y, x, 2) = static_cast<float32>(pixel[2]);   // Red
                } else {
                    throw std::runtime_error("Unsupported number of channels");
                }
            }
        }
    }

    void tensorToMat(const Tensor<float32> &tensor, cv::Mat &dst) {
        if (tensor.empty())
            throw std::runtime_error("Tensor should be non-empty");

        if (tensor.dimensions() == static_cast<size_t>(2))
            dst = cv::Mat(tensor.shape()[0], tensor.shape()[1], CV_8UC1);
        else
            dst = cv::Mat(tensor.shape()[0], tensor.shape()[1], CV_8UC3);

        for (size_t y = 0; y < tensor.shape()[0]; ++y) {
            for (size_t x = 0; x < tensor.shape()[1]; ++x) {
                if (tensor.dimensions() == static_cast<size_t>(2)) {
                    // Grayscale
                    dst.at<uchar>(y, x) = static_cast<uchar>(tensor.at(y, x));
                } else if (tensor.dimensions() == static_cast<size_t>(3)) {
                    // Color
                    cv::Vec3b pixel;
                    for (size_t c = 0; c < tensor.shape()[2]; ++c)
                        pixel[c] = static_cast<uchar>(tensor.at(y, x, c));

                    dst.at<cv::Vec3b>(y, x) = pixel;
                } else {
                    throw std::runtime_error("Unsupported tensor dimensions");
                }
            }
        }
    }

}   // namespace core
