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
        std::cout << "Image size: " << width << " x " << height << std::endl;

        auto [screenWidth, screenHeight] = getScreenResolution();
        std::cout << "Screen size: " << screenWidth << " x " << screenHeight << std::endl;

        const size_t x = (screenWidth - width) / 2;
        const size_t y = (screenHeight - height) / 2;

        imshow(windowName, image);
        cv::moveWindow(windowName, x, y);

        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    void showImageCenterWindow(const Tensor<int> &tensor, const std::string &windowName) {
        cv::Mat image;
        tensorToMat(tensor, image, tensor.shape()[2]);
        showImageCenterWindow(image, windowName);
    }

    void matToTensor(const cv::Mat &src, Tensor<int> &tensor, const size_t channels) {
        if (src.empty())
            throw std::runtime_error("Image should be non-empty");

        if (src.channels() != channels)
            throw std::runtime_error("Image should have the same number of channels");

        tensor = Tensor<int>(src.rows, src.cols, channels);
        for (size_t y = 0; y < src.rows; ++y) {
            for (size_t x = 0; x < src.cols; ++x) {
                const auto &pixel = src.at<cv::Vec3b>(y, x);

                for (size_t c = 0; c < channels; ++c)
                    tensor({y, x, c}) = static_cast<int>(pixel[c]);
            }
        }
    }

    void tensorToMat(const Tensor<int> &tensor, cv::Mat &dst, const size_t channels) {
        if (tensor.empty())
            throw std::runtime_error("Tensor should be non-empty");

        if (tensor.dimensions() != 3 || tensor.shape()[2] != channels)
            throw std::runtime_error(
                "Tensor should have 3 dimensions and the last dimension should match the number of "
                "channels");

        dst = cv::Mat(tensor.shape()[0], tensor.shape()[1], CV_8UC(channels));
        for (size_t y = 0; y < tensor.shape()[0]; ++y) {
            for (size_t x = 0; x < tensor.shape()[1]; ++x) {
                cv::Vec3b pixel;

                for (size_t c = 0; c < channels; ++c)
                    pixel[c] = static_cast<uchar>(tensor({y, x, c}));

                dst.at<cv::Vec3b>(y, x) = pixel;
            }
        }
    }

}   // namespace core