//
// File        : thresholding.cpp
// Author      : Hinsun
// Date        : 2025-06-25
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#ifndef THRESHOLDING_HPP
#define THRESHOLDING_HPP

/**
 * @brief Header for various image thresholding operations.
 *
 * This file declares the `processing::Thresholding` class, which provides static
 * methods for applying different types of thresholding to grayscale images, including:
 * - Simple thresholding (binary, inverse, trunc, etc.)
 * - Adaptive thresholding (mean and Gaussian)
 * - Otsu’s thresholding for automatic threshold selection
 */

#include "core/core.hpp"
#include "core/tensor.hpp"

namespace processing {

class Thresholding {
public:
    /**
     * @enum SimpleThresholding
     * @brief Enum for basic thresholding modes.
     */
    enum class SimpleThresholding {
        THRESH_BINARY,
        THRESH_BINARY_INV,
        THRESH_TRUNC,
        THRESH_TOZERO,
        THRESH_TOZERO_INV
    };

    /**
     * @enum AdaptiveThresholding
     * @brief Enum for adaptive thresholding strategies.
     */
    enum class AdaptiveThresholding { ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C };

    /**
     * @brief Applies simple (global) thresholding to a grayscale image.
     *
     * @param src           Input grayscale image (1 channel).
     * @param dst           Output image after thresholding.
     * @param thresh        Threshold value.
     * @param maxVal        Maximum value to use when thresholding.
     * @param type          Thresholding type to apply (binary, trunc, etc.).
     */
    static void applyThresholding(const core::Tensor<core::float32>& src,
                                  core::Tensor<core::float32>& dst, int thresh, int maxVal,
                                  SimpleThresholding type);

    /**
     * @brief Applies adaptive thresholding to a grayscale image.
     *
     * Computes a threshold value for each pixel based on the local neighborhood.
     *
     * @param src           Input grayscale image (1 channel).
     * @param dst           Output image after adaptive thresholding.
     * @param blockSize     Size of the local region (must be odd and > 1).
     * @param C             Constant subtracted from the mean or weighted mean.
     * @param type          Type of adaptive thresholding (mean or Gaussian).
     */
    static void applyAdaptiveThresholding(const core::Tensor<core::float32>& src,
                                          core::Tensor<core::float32>& dst, int blockSize, double C,
                                          AdaptiveThresholding type);

    /**
     * @brief Applies adaptive thresholding using local mean.
     *
     * For each pixel, computes the mean of its local region and subtracts C to get the
     * threshold.
     *
     * @param src           Input grayscale image (1 channel).
     * @param dst           Output thresholded image.
     * @param blockSize     Neighborhood size (must be odd and > 1).
     * @param C             Constant subtracted from the mean.
     */
    static void adaptiveThresholdMean(const core::Tensor<core::float32>& src,
                                      core::Tensor<core::float32>& dst, int blockSize, int C);

    /**
     * @brief Applies adaptive thresholding using Gaussian-weighted mean.
     *
     * Similar to mean thresholding, but uses a Gaussian kernel to weight pixels in the
     * local region. Better suited for images with uneven lighting.
     *
     * @param src           Input grayscale image (1 channel).
     * @param dst           Output thresholded image.
     * @param blockSize     Neighborhood size (must be odd and > 1).
     * @param C             Constant subtracted from the weighted mean.
     * @param sigma         Standard deviation of the Gaussian kernel.
     */
    static void adaptiveThresholdGaussian(const core::Tensor<core::float32>& src,
                                          core::Tensor<core::float32>& dst, int blockSize, double C,
                                          double sigma);

    /**
     * @brief Computes a 2D Gaussian kernel value at a specific (x, y) location.
     *
     * Used internally for Gaussian adaptive thresholding.
     *
     * @param x     x-coordinate (relative to center).
     * @param y     y-coordinate (relative to center).
     * @param sigma Standard deviation of the Gaussian distribution.
     * @return Gaussian weight at the given position.
     */
    static double gaussianKernel(int x, int y, double sigma);

    /**
     * @brief Computes Otsu's optimal threshold value for a grayscale image.
     *
     * Automatically finds the threshold that minimizes intra-class variance
     * (i.e., best separates foreground from background).
     *
     * @param src Input grayscale image (1 channel).
     * @return Optimal threshold value found by Otsu’s method.
     */
    static int otsuThresholding(const core::Tensor<core::float32>& src);
};

}  // namespace processing

#endif  // THRESHOLDING_HPP
