//
// Created by Hinsun on 2025-06-24.
// Copyright (c) 2025 VanHoai. All rights reserved.
//

#include "processing/thresholding.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

namespace processing {

    void Thresholding::applyThresholding(const cv::Mat &inputImage, cv::Mat &outputImage,
                                         const int thresh, const int maxVal,
                                         const SimpleThresholding type) {
        if (inputImage.empty())
            throw std::invalid_argument("Input image is not empty");

        if (inputImage.channels() != 1)
            throw std::invalid_argument("Input image must be in grayscale format (1 channel)");

        const int width = inputImage.cols;
        const int height = inputImage.rows;

        outputImage = cv::Mat::zeros(height, width, CV_8UC1);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (type == SimpleThresholding::THRESH_BINARY) {
                    if (inputImage.at<uchar>(y, x) >= thresh)
                        outputImage.at<uchar>(y, x) = maxVal;
                    else
                        outputImage.at<uchar>(y, x) = 0;
                } else if (type == SimpleThresholding::THRESH_BINARY_INV) {
                    if (inputImage.at<uchar>(y, x) < thresh)
                        outputImage.at<uchar>(y, x) = maxVal;
                    else
                        outputImage.at<uchar>(y, x) = 0;
                } else if (type == SimpleThresholding::THRESH_TRUNC) {
                    if (inputImage.at<uchar>(y, x) > thresh)
                        outputImage.at<uchar>(y, x) = thresh;
                    else
                        outputImage.at<uchar>(y, x) = inputImage.at<uchar>(y, x);
                } else if (type == SimpleThresholding::THRESH_TOZERO) {
                    if (inputImage.at<uchar>(y, x) > thresh)
                        outputImage.at<uchar>(y, x) = inputImage.at<uchar>(y, x);
                    else
                        outputImage.at<uchar>(y, x) = 0;
                } else if (type == SimpleThresholding::THRESH_TOZERO_INV) {
                    if (inputImage.at<uchar>(y, x) <= thresh)
                        outputImage.at<uchar>(y, x) = inputImage.at<uchar>(y, x);
                    else
                        outputImage.at<uchar>(y, x) = 0;
                } else {
                    throw std::invalid_argument("Unsupported thresholding type");
                }
            }
        }
    }

    void Thresholding::applyAdaptiveThresholding(const cv::Mat &inputImage, cv::Mat &outputImage,
                                                 int blockSize, double C,
                                                 AdaptiveThresholding type) {
        if (inputImage.empty())
            throw std::invalid_argument("Input image is not empty");

        if (inputImage.channels() != 1)
            throw std::invalid_argument("Input image must be in grayscale format (1 channel)");

        if (blockSize % 2 == 0 || blockSize <= 1)
            throw std::invalid_argument("Block size must be an odd number greater than 1");

        if (type == AdaptiveThresholding::ADAPTIVE_THRESH_MEAN_C) {
            adaptiveThresholdMean(inputImage, outputImage, blockSize, static_cast<int>(C));
        } else if (type == AdaptiveThresholding::ADAPTIVE_THRESH_GAUSSIAN_C) {
            adaptiveThresholdGaussian(inputImage, outputImage, blockSize, C, 1.0);
        } else {
            throw std::invalid_argument("Unsupported adaptive thresholding type");
        }
    }

    void Thresholding::adaptiveThresholdMean(const cv::Mat &inputImage, cv::Mat &outputImage,
                                             const int blockSize, const int C) {
        int radius = blockSize / 2;
        outputImage = cv::Mat::zeros(inputImage.size(), CV_8UC1);

        for (int y = 0; y < inputImage.rows; y++) {
            for (int x = 0; x < inputImage.cols; x++) {
                int sum = 0;
                int count = 0;

                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        int ny = y + dy;
                        int nx = x + dx;

                        if (ny >= 0 && ny < inputImage.rows && nx >= 0 && nx < inputImage.cols) {
                            sum += inputImage.at<uchar>(ny, nx);
                            count++;
                        }
                    }
                }

                int threshold = sum / count - C;
                outputImage.at<uchar>(y, x) = inputImage.at<uchar>(y, x) > threshold ? 255 : 0;
            }
        }
    }

    void Thresholding::adaptiveThresholdGaussian(const cv::Mat &inputImage, cv::Mat &outputImage,
                                                 const int blockSize, const double C,
                                                 const double sigma) {
        int radius = blockSize / 2;
        outputImage = cv::Mat::zeros(inputImage.size(), CV_8UC1);

        for (int y = 0; y < inputImage.rows; y++) {
            for (int x = 0; x < inputImage.cols; x++) {
                double sum = 0.0;
                double weightSum = 0.0;

                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int ny = y + dy;
                        int nx = x + dx;
                        if (ny >= 0 && ny < inputImage.rows && nx >= 0 && nx < inputImage.cols) {
                            double weight = gaussianKernel(dx, dy, sigma);
                            sum += weight * inputImage.at<uchar>(ny, nx);
                            weightSum += weight;
                        }
                    }
                }

                int threshold = static_cast<int>(sum / weightSum) - C;
                outputImage.at<uchar>(y, x) = inputImage.at<uchar>(y, x) > threshold ? 255 : 0;
            }
        }
    }

    double Thresholding::gaussianKernel(const int x, const int y, const double sigma) {
        return std::exp(-(x * x + y * y) / (2 * sigma * sigma));
    }

    int Thresholding::otsuThresholding(const cv::Mat &inputImage) {
        if (inputImage.empty())
            throw std::invalid_argument("Input image is not empty");

        if (inputImage.channels() != 1)
            throw std::invalid_argument("Input image must be in grayscale format (1 channel)");

        // 1. Calculate the histogram of the input image
        std::vector<int> histogram(256, 0);
        for (int y = 0; y < inputImage.rows; y++) {
            for (int x = 0; x < inputImage.cols; x++) {
                uchar pixel = inputImage.at<uchar>(y, x);
                histogram[pixel]++;
            }
        }

        int total = inputImage.rows * inputImage.cols;

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
            if (weightBackground == 0)
                continue;

            weightForeground = total - weightBackground;
            // If there are no foreground pixels, break the loop
            if (weightForeground == 0)
                break;

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

}   // namespace processing
