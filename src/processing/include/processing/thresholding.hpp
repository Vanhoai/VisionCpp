//
// Created by Hinsun on 2025-06-24.
// Copyright (c) 2025 VanHoai. All rights reserved.
//

#ifndef THRESHOLDING_H
#define THRESHOLDING_H

#include <opencv2/opencv.hpp>

namespace processing {

    /**
     * Thresholding class provides methods for different types of thresholding operations.
     * It includes Otsu's method for automatic thresholding and various types of thresholding
     * operations that can be applied to images.
     * 1. Global Thresholding
     * 2. Binary Thresholding
     * 3. Inverse Binary Thresholding
     * 4. Truncated Thresholding
     * 5. To Zero Thresholding
     * 6. Inverse To Zero Thresholding
     */
    class Thresholding {
        public:
            enum class SimpleThresholding {
                THRESH_BINARY,
                THRESH_BINARY_INV,
                THRESH_TRUNC,
                THRESH_TOZERO,
                THRESH_TOZERO_INV
            };

            enum class AdaptiveThresholding { ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C };

            /**
             * Applies simple thresholding to the input image.
             * @param:
             * inputImage: The input image in grayscale format (1 channel).
             * outputImage: The output image after thresholding.
             * thresh: The threshold value.
             * maxVal: The maximum value to use with the thresholding type.
             * type: The type of thresholding to apply.
             */
            static void applyThresholding(const cv::Mat &inputImage, cv::Mat &outputImage,
                                          int thresh, int maxVal, SimpleThresholding type);

            /**
             * Applies adaptive thresholding to the input image.
             * @param:
             * inputImage: The input image in grayscale format (1 channel).
             * outputImage: The output image after adaptive thresholding.
             * blockSize: Size of the neighborhood area used for adaptive thresholding.
             * C: Constant subtracted from the mean or weighted mean.
             * type: The type of adaptive thresholding to apply.
             *
             * Notice:
             * blockSize must be an odd number greater than 1.
             * C is a constant subtracted from the mean or weighted mean(typically 3 - 10).
             * AdaptiveThresholding type can be either ADAPTIVE_THRESH_MEAN_C or
             * ADAPTIVE_THRESH_GAUSSIAN_C.
             * Additionally, when use ADAPTIVE_THRESH_GAUSSIAN_C, this algorithm also use
             * sigma to control the Gaussian kernel size.
             */
            static void applyAdaptiveThresholding(const cv::Mat &inputImage, cv::Mat &outputImage,
                                                  int blockSize, double C,
                                                  AdaptiveThresholding type);

            /**
             * Adaptive thresholding using mean value.
             * @param:
             * inputImage: The input image in grayscale format (1 channel).
             * outputImage: The output image after adaptive thresholding.
             * blockSize: Size of the neighborhood area used for adaptive thresholding.
             * C: Constant subtracted from the mean.
             *
             * How it works:
             * For each pixel in the input image, the mean of the pixel values in the
             * neighborhood defined by blockSize is calculated, and then the constant C is
             * subtracted from this mean to determine the threshold for that pixel.
             * If the pixel value is greater than the threshold, it is set to maxVal (255),
             * otherwise it is set to 0.
             */
            static void adaptiveThresholdMean(const cv::Mat &inputImage, cv::Mat &outputImage,
                                              int blockSize, int C);

            /**
             * Adaptive thresholding using Gaussian weighted mean.
             * @param:
             * inputImage: The input image in grayscale format (1 channel).
             * outputImage: The output image after adaptive thresholding.
             * blockSize: Size of the neighborhood area used for adaptive thresholding.
             * C: Constant subtracted from the weighted mean.
             * sigma: Standard deviation for the Gaussian kernel.
             *
             * How it works:
             * For each pixel in the input image, a Gaussian weighted mean of the pixel values
             * in the neighborhood defined by blockSize is calculated, and then the constant C
             * is subtracted from this mean to determine the threshold for that pixel.
             * If the pixel value is greater than the threshold, it is set to maxVal (255),
             * otherwise it is set to 0.
             * This method is more effective in handling images with varying lighting conditions
             * as it considers the local pixel distribution with a Gaussian weighting.
             */
            static void adaptiveThresholdGaussian(const cv::Mat &inputImage, cv::Mat &outputImage,
                                                  int blockSize, double C, double sigma);

            /**
             * Calculates the Gaussian kernel value for a given pixel position.
             * @param:
             * x: The x-coordinate of the pixel.
             * y: The y-coordinate of the pixel.
             * sigma: Standard deviation for the Gaussian kernel.
             * @return: The Gaussian kernel value at the specified pixel position.
             */
            static double gaussianKernel(int x, int y, double sigma);

            /**
             * Calculates the Otsu's threshold value for the input image.
             * Otsu's method is used to find an optimal threshold value that separates the
             * foreground and background in a grayscale image.
             * @param:
             * inputImage: The input image in grayscale format (1 channel).
             * @return: The calculated Otsu's threshold value.
             *
             * How it works:
             * 1. The histogram of the input image is computed.
             * 2. The total number of pixels is calculated.
             * 3. The Otsu's method iterates through all possible threshold values,
             *    calculating the between-class variance for each threshold.
             * 4. The threshold value that maximizes the between-class variance is selected
             *    as the optimal threshold.
             * 5. The function returns this optimal threshold value.
             */
            static int otsuThresholding(const cv::Mat &inputImage);
    };

}   // namespace processing

#endif   // THRESHOLDING_H
