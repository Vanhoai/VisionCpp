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

#include <opencv2/opencv.hpp>

namespace processing {
    class Filters {
        public:
            /**
             * @brief Applies a simple box blur to an image.
             *
             * This method uses a normalized box filter to blur the input image.
             *
             * @param src Input image (can be grayscale or color).
             * @param dst Output blurred image.
             * @param ksize Size of the kernel (must be odd).
             */
            static void gaussianBlur(const cv::Mat& src, cv::Mat& dst, int ksize = 5);

            /**
             * @brief Applies a median blur to an image.
             *
             * This method replaces each pixel with the median of its neighborhood.
             *
             * @param src Input image (can be grayscale or color).
             * @param dst Output blurred image.
             * @param ksize Size of the kernel (must be odd).
             */
            static void medianBlur(const cv::Mat& src, cv::Mat& dst, int ksize = 5);

            /**
             * @brief Applies an unsharp mask to sharpen an image.
             *
             * This method enhances edges by subtracting a blurred version of the image.
             *
             * @param src Input image (can be grayscale or color).
             * @param dst Output sharpened image.
             * @param sigma Standard deviation for Gaussian blur.
             * @param alpha Weighting factor for the sharpened effect.
             */
            static void unsharpMask(const cv::Mat& src, cv::Mat& dst, double sigma = 1.0,
                                    double alpha = 1.5);
    };

}   // namespace processing

#endif   // FILTERS_HPP
