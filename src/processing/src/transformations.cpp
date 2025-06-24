//
// Created by Hinsun on 2025-06-18
// Copyright (c) 2025 VanHoai. All rights reserved.
//

#include "processing/transformations.hpp"

#include <opencv2/opencv.hpp>
#include <random>
#include <stdexcept>

namespace processing {

    void Transformations::convertColorSpace(const cv::Mat& inputImage, cv::Mat& outputImage,
                                            ColorSpace colorSpace) {
        if (colorSpace == ColorSpace::BGR_TO_HSV) {
            convertBGRToHSV(inputImage, outputImage);
        } else if (colorSpace == ColorSpace::HSV_TO_BGR) {
            convertHSVtoBGR(inputImage, outputImage);
        } else {
            throw std::invalid_argument("Unsupported color space conversion");
        }
    }

    /**
     * - R' = R / 255.0
     * - G' = G / 255.0
     * - B' = B / 255.0
     *
     * - Cmax = max(R', G', B')
     * - Cmin = min(R', G', B')
     *
     * - Δ = Cmax - Cmin
     *
     * - Δ = 0, then H = 0 (undefined hue)
     * - Cmax = R', then H = 60 * ((G' - B') / Δ) % 360
     * - Cmax = G', then H = 60 * ((B' - R') / Δ + 2) % 360
     * - Cmax = B', then H = 60 * ((R' - G') / Δ + 4) % 360
     *
     * - Cmax = 0, then S = 0 (undefined saturation)
     * - S = Δ / Cmax
     *
     * - V = Cmax
     */
    void Transformations::convertBGRToHSV(const cv::Mat& inputImage, cv::Mat& outputImage) {
        if (inputImage.empty())
            throw std::invalid_argument("Input image is not empty");

        if (inputImage.channels() != 3)
            throw std::invalid_argument("Input image must be in BGR format (3 channels)");

        outputImage = cv::Mat(inputImage.size(), CV_8UC3);
        for (int i = 0; i < inputImage.rows; i++) {
            for (int j = 0; j < inputImage.cols; j++) {
                const auto& pixel = inputImage.at<cv::Vec3b>(i, j);

                float B = pixel[0] / 255.0f;   // Blue
                float G = pixel[1] / 255.0f;   // Green
                float R = pixel[2] / 255.0f;   // Red

                float Cmax = std::max({R, G, B});
                float Cmin = std::min({R, G, B});
                float delta = Cmax - Cmin;

                float H, S, V;
                if (delta == 0) {
                    H = 0;   // Undefined hue
                } else if (Cmax == R) {
                    H = fmod((60 * ((G - B) / delta)), 360);
                } else if (Cmax == G) {
                    H = fmod((60 * ((B - R) / delta + 2)), 360);
                } else {   // Cmax == B
                    H = fmod((60 * ((R - G) / delta + 4)), 360);
                }

                if (Cmax == 0) {
                    S = 0;   // Undefined saturation
                } else {
                    S = delta / Cmax;
                }

                V = Cmax;

                outputImage.at<cv::Vec3b>(i, j) =
                    cv::Vec3b(static_cast<uchar>(H * 255 / 360),   // Hue
                              static_cast<uchar>(S * 255),         // Saturation
                              static_cast<uchar>(V * 255)          // Value
                    );
            }
        }
    }

    /**
     * Conditions required:
     * - Hue (H) must be in the range [0, 360]
     * - Saturation (S) must be in the range [0, 1]
     * - Value (V) must be in the range [0, 1]
     *
     * 1. Calculate the chroma (C):
     * - C = V * S
     * - X = C * (1 - |(H / 60) % 2 - 1|)
     * - m = V - C
     *
     * 2. Determine the RGB prime values (R', G', B') with H in 6 cases:
     * - [0, 60): (R', G', B') = (C, X, 0)
     * - [60, 120): (R', G', B') = (X, C, 0)
     * - [120, 180): (R', G', B') = (0, C, X)
     * - [180, 240): (R', G', B') = (0, X, C)
     * - [240, 300): (R', G', B') = (X, 0, C)
     * - [300, 360): (R', G', B') = (C, 0, X)
     *
     * 3. Convert the RGB prime values to BGR:
     * - B = (B' + m) * 255
     * - G = (G' + m) * 255
     * - R = (R' + m) * 255
     */
    void Transformations::convertHSVtoBGR(const cv::Mat& inputImage, cv::Mat& outputImage) {
        if (inputImage.empty())
            throw std::invalid_argument("Input image is not empty");

        if (inputImage.channels() != 3)
            throw std::invalid_argument("Input image must be in HSV format (3 channels)");

        outputImage = cv::Mat(inputImage.size(), CV_8UC3);
        for (int i = 0; i < inputImage.rows; i++) {
            for (int j = 0; j < inputImage.cols; j++) {
                const auto& pixel = inputImage.at<cv::Vec3b>(i, j);

                float H = pixel[0] * 360 / 255.0f;   // Hue
                float S = pixel[1] / 255.0f;         // Saturation
                float V = pixel[2] / 255.0f;         // Value

                float C = V * S;
                float X = C * (1 - fabs(fmod(H / 60.0f, 2) - 1));
                float m = V - C;

                float R_prime, G_prime, B_prime;

                if (H < 60) {
                    R_prime = C;
                    G_prime = X;
                    B_prime = 0;
                } else if (H < 120) {
                    R_prime = X;
                    G_prime = C;
                    B_prime = 0;
                } else if (H < 180) {
                    R_prime = 0;
                    G_prime = C;
                    B_prime = X;
                } else if (H < 240) {
                    R_prime = 0;
                    G_prime = X;
                    B_prime = C;
                } else if (H < 300) {
                    R_prime = X;
                    G_prime = 0;
                    B_prime = C;
                } else {
                    R_prime = C;
                    G_prime = 0;
                    B_prime = X;
                }

                auto R = static_cast<uchar>((R_prime + m) * 255);
                auto G = static_cast<uchar>((G_prime + m) * 255);
                auto B = static_cast<uchar>((B_prime + m) * 255);
                outputImage.at<cv::Vec3b>(i, j) = cv::Vec3b(B, G, R);
            }
        }
    }

    void Transformations::convertToGrayScale(const cv::Mat& inputImage, cv::Mat& outputImage) {
        // inputImage: BGR format like OpenCV saved
        if (inputImage.empty())
            throw std::invalid_argument("Input image is not empty");

        if (inputImage.channels() != 3)
            throw std::invalid_argument("Input image must be in BGR format (3 channels)");

        // Convert to grayscale using the formula
        // F(R, G, B) = 0.299 * R + 0.587 * G + 0.114 * B

        outputImage = cv::Mat(inputImage.size(), CV_8UC1);
        for (int i = 0; i < inputImage.rows; ++i) {
            for (int j = 0; j < inputImage.cols; ++j) {
                const auto& pixel = inputImage.at<cv::Vec3b>(i, j);
                auto grayValue = static_cast<uchar>(0.299 * pixel[2] +   // R
                                                    0.587 * pixel[1] +   // G
                                                    0.114 * pixel[0]     // B
                );

                outputImage.at<uchar>(i, j) = grayValue;
            }
        }
    }

    void Transformations::resize(const cv::Mat& inputImage, cv::Mat& outputImage,
                                 const cv::Size& newSize) {
        if (inputImage.empty())
            throw std::invalid_argument("Input image is not empty");

        if (newSize.width <= 0 || newSize.height <= 0)
            throw std::invalid_argument("New size must be positive");

        outputImage = cv::Mat(newSize, inputImage.type());

        // calculate scale factors
        double scaleX = static_cast<double>(inputImage.cols) / newSize.width;
        double scaleY = static_cast<double>(inputImage.rows) / newSize.height;

        for (int y = 0; y < newSize.height; y++) {
            for (int x = 0; x < newSize.width; x++) {
                // calculate the corresponding pixel in the input image
                int srcX = static_cast<int>(x * scaleX);
                int srcY = static_cast<int>(y * scaleY);

                // ensure srcX and srcY are within bounds
                if (srcX >= inputImage.cols)
                    srcX = inputImage.cols - 1;
                if (srcY >= inputImage.rows)
                    srcY = inputImage.rows - 1;

                outputImage.at<cv::Vec3b>(y, x) = inputImage.at<cv::Vec3b>(srcY, srcX);
            }
        }
    }

    void Transformations::normalize(const cv::Mat& inputImage, cv::Mat& outputImage,
                                    const cv::Scalar& mean, const cv::Scalar& std) {
        if (inputImage.empty())
            throw std::invalid_argument("Input image is not empty");

        outputImage = inputImage.clone();
        for (int y = 0; y < inputImage.rows; y++) {
            for (int x = 0; x < inputImage.cols; x++) {
                for (int c = 0; c < inputImage.channels(); c++) {
                    // Normalize each channel
                    uchar pixel = inputImage.at<cv::Vec3b>(y, x)[c];

                    // Apply normalization formula: (pixel - mean) / std
                    double v = (pixel - mean[c]) / std[c];
                    outputImage.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(v);
                }
            }
        }
    }

    void Transformations::pad(const cv::Mat& inputImage, cv::Mat& outputImage,
                              const cv::Size& paddingSize, const cv::Scalar& value) {
        if (inputImage.empty())
            throw std::invalid_argument("Input image is not empty");

        int top = paddingSize.height;
        int bottom = paddingSize.height;
        int left = paddingSize.width;
        int right = paddingSize.width;

        outputImage = cv::Mat(inputImage.rows + top + bottom, inputImage.cols + left + right,
                              inputImage.type(), value);

        for (int y = 0; y < inputImage.rows; y++) {
            for (int x = 0; x < inputImage.cols; x++) {
                outputImage.at<cv::Vec3b>(y + top, x + left) = inputImage.at<cv::Vec3b>(y, x);
            }
        }
    }

    void Transformations::crop(const cv::Mat& inputImage, cv::Mat& outputImage,
                               const cv::Rect& roi) {
        if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > inputImage.cols ||
            roi.y + roi.height > inputImage.rows)
            throw std::invalid_argument("ROI is out of bounds of the input image");

        outputImage = cv::Mat(roi.height, roi.width, inputImage.type());
        for (int y = 0; y < outputImage.rows; y++) {
            for (int x = 0; x < outputImage.cols; x++) {
                int srcX = roi.x + x;
                int srcY = roi.y + y;

                if (srcX >= inputImage.cols || srcY >= inputImage.rows)
                    throw std::out_of_range("Source coordinates are out of bounds");

                outputImage.at<cv::Vec3b>(y, x) = inputImage.at<cv::Vec3b>(srcY, srcX);
            }
        }
    }

    void Transformations::randomCrop(const cv::Mat& inputImage, cv::Mat& outputImage,
                                     const cv::Size& cropSize) {
        if (inputImage.empty())
            throw std::invalid_argument("Input image is not empty");

        if (cropSize.width > inputImage.cols || cropSize.height > inputImage.rows)
            throw std::invalid_argument("Crop size must be smaller than input image size");

        int maxX = inputImage.cols - cropSize.width;
        int maxY = inputImage.rows - cropSize.height;

        std::random_device rd;
        std::mt19937 gen(rd());

        std::uniform_int_distribution<> disX(0, maxX);   // random [0, maxX]
        std::uniform_int_distribution<> disY(0, maxY);   // random [0, maxY]

        int x = disX(gen);
        int y = disY(gen);

        crop(inputImage, outputImage, cv::Rect(x, y, cropSize.width, cropSize.height));
    }

    void Transformations::rotate(const cv::Mat& inputImage, cv::Mat& outputImage,
                                 RotateAngle angle) {
        if (inputImage.empty())
            throw std::invalid_argument("Input image is empty");

        if (angle == RotateAngle::CLOCKWISE_90) {
            outputImage = cv::Mat(inputImage.cols, inputImage.rows, inputImage.type());
            for (int y = 0; y < inputImage.rows; ++y)
                for (int x = 0; x < inputImage.cols; ++x)
                    outputImage.at<cv::Vec3b>(x, inputImage.rows - y - 1) =
                        inputImage.at<cv::Vec3b>(y, x);
        } else if (angle == RotateAngle::CLOCKWISE_180) {
            outputImage = inputImage.clone();
            for (int y = 0; y < inputImage.rows; ++y)
                for (int x = 0; x < inputImage.cols; ++x)
                    outputImage.at<cv::Vec3b>(inputImage.rows - y - 1, inputImage.cols - x - 1) =
                        inputImage.at<cv::Vec3b>(y, x);
        } else if (angle == RotateAngle::CLOCKWISE_270) {
            outputImage = cv::Mat(inputImage.cols, inputImage.rows, inputImage.type());
            for (int y = 0; y < inputImage.rows; ++y)
                for (int x = 0; x < inputImage.cols; ++x)
                    outputImage.at<cv::Vec3b>(inputImage.cols - x - 1, y) =
                        inputImage.at<cv::Vec3b>(y, x);
        } else {
            throw std::invalid_argument("Only 90, 180, 270 degrees supported for manual rotation");
        }
    }

    void Transformations::flip(const cv::Mat& inputImage, cv::Mat& outputImage, FlipCode flipCode) {
        if (inputImage.empty())
            throw std::invalid_argument("Input image is empty");

        outputImage = inputImage.clone();
        for (int y = 0; y < outputImage.rows; y++) {
            for (int x = 0; x < outputImage.cols; x++) {
                int newX = (flipCode == FlipCode::HORIZONTAL || flipCode == FlipCode::BOTH)
                               ? inputImage.cols - x - 1
                               : x;
                int newY = (flipCode == FlipCode::VERTICAL || flipCode == FlipCode::BOTH)
                               ? inputImage.rows - y - 1
                               : y;

                outputImage.at<cv::Vec3b>(newY, newX) = inputImage.at<cv::Vec3b>(y, x);
            }
        }
    }

}   // namespace processing
