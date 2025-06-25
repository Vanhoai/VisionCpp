//
// File        : common.hpp
// Author      : Hinsun
// Date        : 2025-06-19
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#ifndef COMMON_HPP
#define COMMON_HPP

#include <core/tensor.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace core {

    std::pair<size_t, size_t> getScreenResolution();
    void showImageCenterWindow(const cv::Mat &image, const std::string &windowName = "Window");
    void showImageCenterWindow(const Tensor<int> &tensor, const std::string &windowName = "Window");

    void matToTensor(const cv::Mat &src, Tensor<int> &tensor, size_t channels = 3);
    void tensorToMat(const Tensor<int> &tensor, cv::Mat &dst, size_t channels = 3);
}   // namespace core

#endif   // COMMON_HPP
