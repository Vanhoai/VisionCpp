//
// File        : operators.cpp
// Author      : Hinsun
// Date        : 2025-06-TODAY
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#include "processing/operators.hpp"

#include <algorithm>

#include "core/core.hpp"

namespace processing {

core::TensorF32 convolveHorizontal(const core::TensorF32& src,
                                   const std::vector<core::float32>& kernel) {
    const std::vector<size_t>& shape = src.shape();
    const size_t height = shape[0];
    const size_t width = shape[1];

    const int ksize = static_cast<int>(kernel.size());
    const int khalf = ksize / 2;

    core::TensorF32 dst(height, width);
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            core::float32 sum = 0.0f;

            for (int k = -khalf; k <= khalf; k++) {
                int xx = std::clamp(static_cast<int>(x) + k, 0, static_cast<int>(width) - 1);
                sum += src.at(y, xx) * kernel[k + khalf];
            }

            dst.at(y, x) = sum;
        }
    }

    return dst;
}

core::TensorF32 convolveVertical(const core::TensorF32& src,
                                 const std::vector<core::float32>& kernel) {
    const std::vector<size_t>& shape = src.shape();
    const size_t height = shape[0];
    const size_t width = shape[1];

    const int ksize = static_cast<int>(kernel.size());
    const int khalf = ksize / 2;

    core::TensorF32 dst(height, width);
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            core::float32 sum = 0.0f;

            for (int k = -khalf; k <= khalf; k++) {
                int yy = std::clamp(static_cast<int>(y) + k, 0, static_cast<int>(height) - 1);
                sum += src.at(yy, x) * kernel[k + khalf];
            }

            dst.at(y, x) = sum;
        }
    }

    return dst;
}

core::TensorF32 gaussianBlur(const core::TensorF32& src, core::float32 sigma) {
    int kernelSize = static_cast<int>(6 * sigma) | 1;  // Ensure odd size
    std::vector<core::float32> kernel(kernelSize);

    float sum = 0.0f;
    int half = kernelSize / 2;

    for (int i = 0; i < kernelSize; i++) {
        float x = i - half;
        kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }

    // Normalize kernel
    for (float& val : kernel) val /= sum;

    // Apply separable convolution (horizontal then vertical)
    core::TensorF32 temp = convolveHorizontal(src, kernel);
    return convolveVertical(temp, kernel);
}

core::TensorF32 downsample(const core::TensorF32& src) {
    const std::vector<size_t>& shape = src.shape();
    const size_t height = shape[0];
    const size_t width = shape[1];

    const size_t newHeight = height / 2;
    const size_t newWidth = width / 2;

    core::TensorF32 dst(newHeight, newWidth);
    for (size_t y = 0; y < newHeight; y++) {
        for (size_t x = 0; x < newWidth; x++) dst.at(y, x) = src.at(y * 2, x * 2);
    }

    return dst;
}

core::TensorF32 substract(const core::TensorF32& a, const core::TensorF32& b) {
    if (a.shape() != b.shape())
        throw std::invalid_argument("Tensors must have the same shape for subtraction.");

    core::TensorF32 substracted(a.shape());
    for (size_t i = 0; i < a.size(); i++) substracted.data()[i] = a.data()[i] - b.data()[i];
    return substracted;
}

}  // namespace processing
