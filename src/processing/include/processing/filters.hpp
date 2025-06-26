//
// File        : filters.hpp
// Author      : Hinsun
// Date        : 2025-06-24
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#ifndef FILTERS_HPP
#define FILTERS_HPP

/**
 * @brief Header for various image filtering operations.
 *
 * This file declares the `processing::Filters` class, which provides static
 * methods for applying different types of filters to images, including:
 * - Blur - Gaussian Blur, Median Blur
 * - Sharpen - Unsharp Masking, Bilateral Filter
 * - Edge Detection - Sobel, Canny
 */

#include "core/core.hpp"
#include "core/tensor.hpp"

namespace processing {
    /**
     * @enum SobelDirection
     * @brief Enum for specifying the direction of Sobel filter application.
     *
     * This enum is used to indicate whether to compute the Sobel filter in the X, Y, or both
     * directions.
     */
    enum class SobelDirection { X, Y, XY };

    constexpr int kernelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    constexpr int kernelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    class Filters {
        public:
            /**
             * @brief Applies a simple box blur to an image.
             *
             * This method uses a normalized box filter to blur the input image.
             *
             * @param src   Input image (can be grayscale or color).
             * @param dst   Output blurred image.
             * @param ksize Size of the kernel (must be odd).
             */
            static void gaussianBlur(const core::Tensor<core::float32>& src,
                                     core::Tensor<core::float32>& dst, int ksize = 5);

            /**
             * @brief Applies a median blur to an image.
             *
             * This method replaces each pixel with the median of its neighborhood.
             *
             * @param src   Input image (can be grayscale or color).
             * @param dst   Output blurred image.
             * @param ksize Size of the kernel (must be odd).
             */
            static void medianBlur(const core::Tensor<core::float32>& src,
                                   core::Tensor<core::float32>& dst, int ksize = 5);

            /**
             * @brief Applies an unsharp mask to sharpen an image.
             *
             * This method enhances edges by subtracting a blurred version of the image.
             *
             * @param src   Input image (can be grayscale or color).
             * @param dst   Output sharpened image.
             * @param sigma Standard deviation for Gaussian blur.
             * @param alpha Weighting factor for the sharpened effect.
             */
            static void unsharpMask(const core::Tensor<core::float32>& src,
                                    core::Tensor<core::float32>& dst, double sigma = 1.0,
                                    double alpha = 1.5);

            /**
             * @brief Applies a bilateral filter to smooth an image while preserving edges.
             *
             * This method uses spatial and range Gaussian functions to filter the image.
             *
             * @param src           Input image (can be grayscale or color).
             * @param dst           Output filtered image.
             * @param ksize         Size of the kernel (must be odd).
             * @param sigmaSpatial  Standard deviation for spatial Gaussian.
             * @param sigmaRange    Standard deviation for range Gaussian.
             *
             * How it works:
             * 1. For each pixel, it considers a neighborhood defined by `ksize`.
             * 2. It computes a spatial weight based on the distance from the center pixel.
             * 3. It computes a range weight based on the intensity difference from the center
             * pixel.
             * 4. The final pixel value is a weighted average of the neighborhood pixels,
             */
            static void bilateralFilter(const core::Tensor<core::float32>& src,
                                        core::Tensor<core::float32>& dst, int ksize = 5,
                                        double sigmaSpatial = 75.0, double sigmaRange = 75.0);

            /**
             * @brief Applies Sobel edge detection to an image.
             *
             * This method computes the gradient magnitude and direction using the Sobel operator.
             *
             * @param src       Input image (can be grayscale or color).
             * @param gradX     Output gradient in the X direction.
             * @param gradY     Output gradient in the Y direction.
             * @param direction Direction of Sobel filter application (X, Y, or both).
             */
            static void sobel(const core::Tensor<core::float32>& src,
                              core::Tensor<core::float32>& gradX,
                              core::Tensor<core::float32>& gradY, SobelDirection direction);

            /**
             * @brief Computes the magnitude and direction of gradients from Sobel outputs.
             *
             * This method calculates the gradient magnitude and direction from the Sobel
             * gradients.
             *
             * @param gradX      Gradient in the X direction.
             * @param gradY      Gradient in the Y direction.
             * @param magnitude  Output gradient magnitude.
             * @param direction  Output gradient direction (angle).
             */
            static void computeMagnitudeDirection(const core::Tensor<core::float32>& gradX,
                                                  const core::Tensor<core::float32>& gradY,
                                                  core::Tensor<core::float32>& magnitude,
                                                  core::Tensor<core::float32>& direction);

            /**
             * @brief Applies Canny edge detection to an image.
             *
             * This method performs Canny edge detection, which includes:
             * 1. Noise Reduction:
             *    - Applies a Gaussian blur (e.g., 5×5 kernel with σ ≈ 1.4) to reduce image noise
             * and detail.
             *    - Converts the image to grayscale to ensure single-channel processing.
             *
             * 2. Gradient Computation:
             *    - Computes image gradients in the X and Y directions using 3×3 Sobel operators.
             *    - Calculates gradient magnitude using the formula: sqrt(Gx² + Gy²).
             *    - Calculates gradient direction using: atan2(Gy, Gx), and normalizes the angle to
             * [0°, 180°).
             *
             * 3. Non-Maximum Suppression:
             *    - Thins the edges by preserving only local maxima in the direction of the
             * gradient.
             *    - Compares each pixel with its two neighbors along the gradient direction.
             *    - Approximates gradient direction to four discrete angles: 0°, 45°, 90°, and 135°.
             *    - Suppresses pixels that are not local maxima.
             *
             * 4. Hysteresis Thresholding:
             *    - Applies two thresholds: high (strong edge) and low (weak edge).
             *    - Marks pixels with gradient magnitude ≥ high threshold as strong edges (value =
             * 255).
             *    - Marks pixels between thresholds as weak edges (value = 128).
             *    - Tracks and retains weak edges connected to strong edges using DFS or BFS.
             *    - Discards all other weak edges as noise.
             *
             * @param src           Input image (can be grayscale or color).
             * @param dst           Output image with edges detected.
             * @param lowThreshold  Low threshold for hysteresis.
             * @param highThreshold High threshold for hysteresis.
             * @param ksize         Size of the Gaussian kernel (must be odd).
             */
            static void canny(const core::Tensor<core::float32>& src,
                              core::Tensor<core::float32>& dst, double lowThreshold,
                              double highThreshold, int ksize);
    };

}   // namespace processing

#endif   // FILTERS_HPP
