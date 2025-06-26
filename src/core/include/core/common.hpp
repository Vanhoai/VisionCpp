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
#include <opencv2/opencv.hpp>

#include "core.hpp"

namespace core {

    std::pair<size_t, size_t> getScreenResolution();
    void showImageCenterWindow(const cv::Mat &image, const std::string &windowName = "Window");
    void showImageCenterWindow(const Tensor<float32> &tensor,
                               const std::string &windowName = "Window");

    void matToTensor(const cv::Mat &src, Tensor<float32> &tensor);
    void tensorToMat(const Tensor<float32> &tensor, cv::Mat &dst);
}   // namespace core

#endif   // COMMON_HPP
