//
// Created by Hinsun on 2025-06-24.
// Copyright (c) 2025 VanHoai. All rights reserved.
//

#include "processing/thresholding.hpp"

#include <vector>

#include "core/core.hpp"
#include "core/tensor.hpp"

namespace processing {

void Thresholding::applyThresholding(const core::Tensor<core::float32> &src,
                                     core::Tensor<core::float32> &dst, const int thresh,
                                     const int maxVal, const SimpleThresholding type) {
    if (src.empty()) throw std::invalid_argument("Input image is not empty");

    if (src.dimensions() != 2)
        throw std::invalid_argument("Input image must be in grayscale format (1 channel)");

    const int width = src.shape()[0];
    const int height = src.shape()[1];

    dst = core::Tensor<core::float32>(height, width);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (type == SimpleThresholding::THRESH_BINARY) {
                if (src.at(y, x) >= thresh)
                    dst.at(y, x) = maxVal;
                else
                    dst.at(y, x) = 0;
            } else if (type == SimpleThresholding::THRESH_BINARY_INV) {
                if (src.at(y, x) < thresh)
                    dst.at(y, x) = maxVal;
                else
                    dst.at(y, x) = 0;
            } else if (type == SimpleThresholding::THRESH_TRUNC) {
                if (src.at(y, x) > thresh)
                    dst.at(y, x) = thresh;
                else
                    dst.at(y, x) = src.at(y, x);
            } else if (type == SimpleThresholding::THRESH_TOZERO) {
                if (src.at(y, x) > thresh)
                    dst.at(y, x) = src.at(y, x);
                else
                    dst.at(y, x) = 0;
            } else if (type == SimpleThresholding::THRESH_TOZERO_INV) {
                if (src.at(y, x) <= thresh)
                    dst.at(y, x) = src.at(y, x);
                else
                    dst.at(y, x) = 0;
            } else {
                throw std::invalid_argument("Unsupported thresholding type");
            }
        }
    }
}

void Thresholding::applyAdaptiveThresholding(const core::Tensor<core::float32> &src,
                                             core::Tensor<core::float32> &dst, int blockSize,
                                             double C, AdaptiveThresholding type) {
    if (src.empty()) throw std::invalid_argument("Input image is not empty");

    if (src.dimensions() != 2)
        throw std::invalid_argument("Input image must be in grayscale format (1 channel)");

    if (blockSize % 2 == 0 || blockSize <= 1)
        throw std::invalid_argument("Block size must be an odd number greater than 1");

    if (type == AdaptiveThresholding::ADAPTIVE_THRESH_MEAN_C) {
        adaptiveThresholdMean(src, dst, blockSize, static_cast<int>(C));
    } else if (type == AdaptiveThresholding::ADAPTIVE_THRESH_GAUSSIAN_C) {
        adaptiveThresholdGaussian(src, dst, blockSize, C, 1.0);
    } else {
        throw std::invalid_argument("Unsupported adaptive thresholding type");
    }
}

void Thresholding::adaptiveThresholdMean(const core::Tensor<core::float32> &src,
                                         core::Tensor<core::float32> &dst, const int blockSize,
                                         const int C) {
    const int radius = blockSize / 2;
    const int height = src.shape()[0];
    const int width = src.shape()[1];

    dst = core::Tensor<core::float32>(height, width);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int sum = 0;
            int count = 0;

            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    const int ny = y + dy;
                    const int nx = x + dx;

                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        sum += src.at(ny, nx);
                        count++;
                    }
                }
            }

            const int threshold = sum / count - C;
            dst.at(y, x) = src.at(y, x) > threshold ? 255 : 0;
        }
    }
}

void Thresholding::adaptiveThresholdGaussian(const core::Tensor<core::float32> &src,
                                             core::Tensor<core::float32> &dst, const int blockSize,
                                             const double C, const double sigma) {
    const int radius = blockSize / 2;
    const int height = src.shape()[0];
    const int width = src.shape()[1];

    dst = core::Tensor<core::float32>(height, width);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double sum = 0.0;
            double weightSum = 0.0;

            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int ny = y + dy;
                    const int nx = x + dx;
                    if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                        const double weight = gaussianKernel(dx, dy, sigma);
                        sum += weight * src.at(ny, nx);
                        weightSum += weight;
                    }
                }
            }

            const int threshold = static_cast<int>(sum / weightSum) - C;
            dst.at(y, x) = src.at(y, x) > threshold ? 255 : 0;
        }
    }
}

double Thresholding::gaussianKernel(const int x, const int y, const double sigma) {
    return std::exp(-(x * x + y * y) / (2 * sigma * sigma));
}

int Thresholding::otsuThresholding(const core::Tensor<core::float32> &src) {
    if (src.empty()) throw std::invalid_argument("Input image is not empty");

    if (src.dimensions() != 2)
        throw std::invalid_argument("Input image must be in grayscale format (1 channel)");

    const int height = src.shape()[0];
    const int width = src.shape()[1];

    // 1. Calculate the histogram of the input image
    std::vector histogram(256, 0);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int pixel = static_cast<int>(src.at(y, x));
            histogram[pixel]++;
        }
    }

    int total = height * width;

    // 2. Calculate the cumulative histogram and the total number of pixels
    double sum = 0.0;
    for (int i = 0; i < 255; i++) sum += i * histogram[i];

    double sumBackground = 0.0;
    int weightBackground = 0;
    int weightForeground = 0;

    double maxVariance = 0.0;
    int optimalThreshold = 0;

    for (int t = 0; t < 256; t++) {
        weightBackground += histogram[t];
        if (weightBackground == 0) continue;

        weightForeground = total - weightBackground;
        // If there are no foreground pixels, break the loop
        if (weightForeground == 0) break;

        sumBackground += t * histogram[t];
        double meanBackground = sumBackground / weightBackground;
        double meanForeground = (sum - sumBackground) / weightForeground;

        // Calculate the between-class variance
        double variance = static_cast<double>(weightBackground) *
                          static_cast<double>(weightForeground) *
                          (meanBackground - meanForeground) * (meanBackground - meanForeground);

        // Update the maximum variance and optimal threshold
        if (variance > maxVariance) {
            maxVariance = variance;
            optimalThreshold = t;
        }
    }

    return optimalThreshold;
}

}  // namespace processing
