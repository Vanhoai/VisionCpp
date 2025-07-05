//
// Created by Hinsun on 2025-06-18
// Copyright (c) 2025 VanHoai. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <processing/transformations.hpp>
#include <random>
#include <stdexcept>

#include "core/core.hpp"

namespace processing {

void Transformations::convertColorSpace(const core::TensorF32& src, core::TensorF32& dst,
                                        ColorSpace colorSpace) {
    if (colorSpace == ColorSpace::BGR_TO_HSV) {
        convertBGRToHSV(src, dst);
    } else if (colorSpace == ColorSpace::HSV_TO_BGR) {
        convertHSVtoBGR(src, dst);
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
void Transformations::convertBGRToHSV(const core::TensorF32& src, core::TensorF32& dst) {
    if (src.empty()) throw std::invalid_argument("Input image is not empty");

    if (src.shape().size() != 3)
        throw std::invalid_argument("Input image must have 3 dimensions (height, width, channels)");

    if (src.shape()[2] != 3)
        throw std::invalid_argument("Input image must be in BGR format (3 channels)");

    dst = core::TensorF32(src.shape());

    for (size_t i = 0; i < src.shape()[0]; i++) {
        for (size_t j = 0; j < src.shape()[1]; j++) {
            float B = src.at(i, j, 0) / 255.0f;  // Blue
            float G = src.at(i, j, 1) / 255.0f;  // Green
            float R = src.at(i, j, 2) / 255.0f;  // Red

            const float Cmax = std::max({R, G, B});
            const float Cmin = std::min({R, G, B});
            const float delta = Cmax - Cmin;

            float H, S;
            if (delta == 0) {
                H = 0;  // Undefined hue
            } else if (Cmax == R) {
                H = fmod((60 * ((G - B) / delta)), 360);
            } else if (Cmax == G) {
                H = fmod((60 * ((B - R) / delta + 2)), 360);
            } else {  // Cmax == B
                H = fmod((60 * ((R - G) / delta + 4)), 360);
            }

            if (Cmax == 0) {
                S = 0;  // Undefined saturation
            } else {
                S = delta / Cmax;
            }

            const float V = Cmax;
            dst.at(i, j, 0) = H * 255 / 360;  // Hue in [0, 360] scaled to [0, 255]
            dst.at(i, j, 1) = S * 255;        // Saturation in [0, 1] scaled to [0, 255]
            dst.at(i, j, 2) = V * 255;        // Value in [0, 1] scaled to [0, 255]
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
void Transformations::convertHSVtoBGR(const core::TensorF32& src, core::TensorF32& dst) {
    if (src.empty()) throw std::invalid_argument("Input image is not empty");

    if (src.shape().size() != 3)
        throw std::invalid_argument("Input image must have 3 dimensions (height, width, channels)");

    if (src.shape()[2] != 3)
        throw std::invalid_argument("Input image must be in HSV format (3 channels)");

    dst = core::TensorF32(src.shape());

    for (size_t i = 0; i < src.shape()[0]; i++) {
        for (size_t j = 0; j < src.shape()[1]; j++) {
            const float H = src.at(i, j, 0) * 360 / 255.0f;  // Hue
            const float S = src.at(i, j, 1) / 255.0f;        // Saturation
            const float V = src.at(i, j, 2) / 255.0f;        // Value

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

            const auto R = static_cast<uchar>((R_prime + m) * 255);
            const auto G = static_cast<uchar>((G_prime + m) * 255);
            const auto B = static_cast<uchar>((B_prime + m) * 255);

            dst.at(i, j, 0) = B;  // Blue
            dst.at(i, j, 1) = G;  // Green
            dst.at(i, j, 2) = R;  // Red
        }
    }
}

void Transformations::convertToGrayScale(const core::TensorF32& src, core::TensorF32& dst) {
    if (src.empty()) throw std::invalid_argument("Input image is not empty");

    if (src.shape().size() != 3)
        throw std::invalid_argument("Input image must have 3 dimensions (height, width, channels)");

    if (src.shape()[2] != 3)
        throw std::invalid_argument("Input image must be in BGR format (3 channels)");

    // Convert to grayscale using the formula
    // F(R, G, B) = 0.299 * R + 0.587 * G + 0.114 * B

    dst = core::TensorF32(src.shape()[0], src.shape()[1]);

    for (size_t i = 0; i < src.shape()[0]; ++i) {
        for (size_t j = 0; j < src.shape()[1]; ++j) {
            const core::float32 grayValue =
                0.299f * src.at(i, j, 2) + 0.587f * src.at(i, j, 1) + 0.114f * src.at(i, j, 0);

            dst.at(i, j) = grayValue;
        }
    }
}

core::TensorF32 Transformations::convertToGrayScale(const core::TensorF32& src) {
    if (src.empty()) throw std::invalid_argument("Input image is not empty");

    if (src.shape().size() != 3)
        throw std::invalid_argument("Input image must have 3 dimensions (height, width, channels)");

    if (src.shape()[2] != 3)
        throw std::invalid_argument("Input image must be in BGR format (3 channels)");

    // Convert to grayscale using the formula
    // F(R, G, B) = 0.299 * R + 0.587 * G + 0.114 * B

    const size_t height = src.shape()[0];
    const size_t width = src.shape()[1];

    core::TensorF32 dst(height, width);
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            const core::float32 grayValue =
                0.299f * src.at(y, x, 2) + 0.587f * src.at(y, x, 1) + 0.114f * src.at(y, x, 0);

            dst.at(y, x) = grayValue;
        }
    }

    return dst;
}

void Transformations::resize(const core::TensorF32& src, core::TensorF32& dst, const size_t width,
                             const size_t height) {
    if (src.empty()) throw std::invalid_argument("Input image is not empty");

    if (width <= 0 || height <= 0) throw std::invalid_argument("New size must be positive");

    dst = core::TensorF32(height, width, src.shape()[2]);

    // calculate scale factors
    const double scaleY = static_cast<double>(src.shape()[0]) / height;
    const double scaleX = static_cast<double>(src.shape()[1]) / width;

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            // calculate the corresponding pixel in the input image
            auto srcX = static_cast<size_t>(x * scaleX);
            auto srcY = static_cast<size_t>(y * scaleY);

            // ensure srcX and srcY are within bounds
            if (srcX >= src.shape()[1]) srcX = src.shape()[1] - 1;
            if (srcY >= src.shape()[0]) srcY = src.shape()[0] - 1;

            if (src.shape()[2] == 1) {
                // Case 1: 1 channel
                dst.at(y, x) = src.at(srcY, srcX);
            } else if (src.shape()[2] == 3) {
                // Case 2: 3 channels
                dst.at(y, x, 0) = src.at(srcY, srcX, 0);  // Blue
                dst.at(y, x, 1) = src.at(srcY, srcX, 1);  // Green
                dst.at(y, x, 2) = src.at(srcY, srcX, 2);  // Red
            } else {
                throw std::invalid_argument("Unsupported number of channels for resizing");
            }
        }
    }
}

void Transformations::normalize(const core::TensorF32& src, core::TensorF32& dst,
                                const std::vector<core::float32>& mean,
                                const std::vector<core::float32>& std) {
    if (src.empty()) throw std::invalid_argument("Input image is not empty");

    dst = src.clone();

    for (size_t y = 0; y < src.shape()[0]; y++) {
        for (size_t x = 0; x < src.shape()[1]; x++) {
            for (size_t c = 0; c < src.shape()[2]; c++) {
                // Normalize each channel
                const core::float32 pixel = src.at(y, x, c);

                // Apply normalization formula: (pixel - mean) / std
                const core::float32 v = (pixel - mean[c]) / std[c];
                dst.at(y, x, c) = v;
            }
        }
    }
}

void Transformations::pad(const core::TensorF32& src, core::TensorF32& dst,
                          const core::float32 padding, const core::float32 value) {
    if (src.empty()) throw std::invalid_argument("Input image is not empty");

    const size_t height = src.shape()[0] + padding * 2;
    const size_t width = src.shape()[1] + padding * 2;

    if (src.shape()[2] == 1)
        dst = core::Tensor({height, width}, value);
    else if (src.shape()[2] == 3)
        dst = core::Tensor({height, width, src.shape()[2]}, value);
    else
        throw std::invalid_argument("Unsupported number of channels for padding");

    for (size_t y = 0; y < src.shape()[0]; y++) {
        for (size_t x = 0; x < src.shape()[1]; x++) {
            // Case 1: 1 channels
            if (src.shape()[2] == 1)
                dst.at(y + padding, x + padding) = src.at(y, x);
            else if (src.shape()[2] == 3) {
                // Case 2: 3 channels
                dst.at(y + padding, x + padding, 0) = src.at(y, x, 0);  // Blue
                dst.at(y + padding, x + padding, 1) = src.at(y, x, 1);  // Green
                dst.at(y + padding, x + padding, 2) = src.at(y, x, 2);  // Red
            } else {
                throw std::invalid_argument("Unsupported number of channels for padding");
            }
        }
    }
}

void Transformations::crop(const core::TensorF32& src, core::TensorF32& dst,
                           const core::Rect& roi) {
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > src.shape()[1] ||
        roi.y + roi.height > src.shape()[0])
        throw std::invalid_argument("ROI is out of bounds of the input image");

    if (src.shape()[2] == 1)
        dst = core::TensorF32(roi.height, roi.width, 1);
    else if (src.shape()[2] == 3)
        dst = core::TensorF32(roi.height, roi.width, 3);
    else
        throw std::invalid_argument("Unsupported number of channels for cropping");

    for (size_t y = 0; y < dst.shape()[0]; y++) {
        for (size_t x = 0; x < dst.shape()[1]; x++) {
            const size_t srcX = roi.x + x;
            const size_t srcY = roi.y + y;

            if (srcX >= src.shape()[1] || srcY >= src.shape()[0])
                throw std::out_of_range("Source coordinates are out of bounds");

            if (src.shape()[2] == 1) {
                // Case 1: 1 channel
                dst.at(y, x) = src.at(srcY, srcX);
            } else if (src.shape()[2] == 3) {
                // Case 2: 3 channels
                dst.at(y, x, 0) = src.at(srcY, srcX, 0);  // Blue
                dst.at(y, x, 1) = src.at(srcY, srcX, 1);  // Green
                dst.at(y, x, 2) = src.at(srcY, srcX, 2);  // Red
            } else {
                throw std::invalid_argument("Unsupported number of channels for cropping");
            }
        }
    }
}

void Transformations::randomCrop(const core::TensorF32& src, core::TensorF32& dst,
                                 const size_t width, const size_t height) {
    if (src.empty()) throw std::invalid_argument("Input image is not empty");

    if (width > src.shape()[1] || height > src.shape()[0])
        throw std::invalid_argument("Crop size must be smaller than input image size");

    const int maxX = src.shape()[1] - width;
    const int maxY = src.shape()[0] - height;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> disX(0, maxX);  // random [0, maxX]
    std::uniform_int_distribution<> disY(0, maxY);  // random [0, maxY]

    const int x = disX(gen);
    const int y = disY(gen);

    crop(src, dst, core::Rect(x, y, width, height));
}

void Transformations::rotate(const core::TensorF32& src, core::TensorF32& dst, RotateAngle angle) {
    if (src.empty()) throw std::invalid_argument("Input image is empty");

    if (angle == RotateAngle::CLOCKWISE_90) {
        dst = core::TensorF32(src.shape()[1], src.shape()[0], src.shape()[2]);

        for (size_t y = 0; y < src.shape()[0]; ++y) {
            for (size_t x = 0; x < src.shape()[1]; ++x) {
                // dst.at<cv::Vec3b>(x, src.shape()[0] - y - 1) = src.at<cv::Vec3b>(y, x);

                if (src.shape()[2] == 1) {
                    // Case 1: 1 channel
                    dst.at(x, src.shape()[0] - y - 1) = src.at(y, x);
                } else if (src.shape()[2] == 3) {
                    // Case 2: 3 channels
                    dst.at(x, src.shape()[0] - y - 1, 0) = src.at(y, x, 0);  // Blue
                    dst.at(x, src.shape()[0] - y - 1, 1) = src.at(y, x, 1);  // Green
                    dst.at(x, src.shape()[0] - y - 1, 2) = src.at(y, x, 2);  // Red
                } else {
                    throw std::invalid_argument("Unsupported number of channels for rotation");
                }
            }
        }

    } else if (angle == RotateAngle::CLOCKWISE_180) {
        dst = src.clone();

        for (size_t y = 0; y < src.shape()[0]; ++y) {
            for (size_t x = 0; x < src.shape()[1]; ++x) {
                // dst.at<cv::Vec3b>(src.shape()[0] - y - 1, src.shape()[1] - x - 1) =
                //     src.at<cv::Vec3b>(y, x);

                if (src.shape()[2] == 1) {
                    // Case 1: 1 channel
                    dst.at(src.shape()[0] - y - 1, src.shape()[1] - x - 1) = src.at(y, x);
                } else if (src.shape()[2] == 3) {
                    // Case 2: 3 channels
                    dst.at(src.shape()[0] - y - 1, src.shape()[1] - x - 1, 0) =
                        src.at(y, x, 0);  // Blue
                    dst.at(src.shape()[0] - y - 1, src.shape()[1] - x - 1, 1) =
                        src.at(y, x, 1);  // Green
                    dst.at(src.shape()[0] - y - 1, src.shape()[1] - x - 1, 2) =
                        src.at(y, x, 2);  // Red
                } else {
                    throw std::invalid_argument("Unsupported number of channels for rotation");
                }
            }
        }

    } else if (angle == RotateAngle::CLOCKWISE_270) {
        dst = core::TensorF32(src.shape()[1], src.shape()[0], src.shape()[2]);

        for (size_t y = 0; y < src.shape()[0]; ++y) {
            for (size_t x = 0; x < src.shape()[1]; ++x) {
                // dst.at<cv::Vec3b>(src.shape()[1] - x - 1, y) = src.at<cv::Vec3b>(y, x);

                if (src.shape()[2] == 1) {
                    // Case 1: 1 channel
                    dst.at(src.shape()[1] - x - 1, y) = src.at(y, x);
                } else if (src.shape()[2] == 3) {
                    // Case 2: 3 channels
                    dst.at(src.shape()[1] - x - 1, y, 0) = src.at(y, x, 0);  // Blue
                    dst.at(src.shape()[1] - x - 1, y, 1) = src.at(y, x, 1);  // Green
                    dst.at(src.shape()[1] - x - 1, y, 2) = src.at(y, x, 2);  // Red
                } else {
                    throw std::invalid_argument("Unsupported number of channels for rotation");
                }
            }
        }

    } else {
        throw std::invalid_argument("Only 90, 180, 270 degrees supported for manual rotation");
    }
}

void Transformations::flip(const core::TensorF32& src, core::TensorF32& dst, FlipCode flipCode) {
    if (src.empty()) throw std::invalid_argument("Input image is empty");

    dst = src.clone();
    for (size_t y = 0; y < dst.shape()[0]; y++) {
        for (size_t x = 0; x < dst.shape()[1]; x++) {
            const int newX = (flipCode == FlipCode::HORIZONTAL || flipCode == FlipCode::BOTH)
                                 ? src.shape()[1] - x - 1
                                 : x;
            const int newY = (flipCode == FlipCode::VERTICAL || flipCode == FlipCode::BOTH)
                                 ? src.shape()[0] - y - 1
                                 : y;

            // dst.at<cv::Vec3b>(newY, newX) = src.at<cv::Vec3b>(y, x);
            if (src.shape()[2] == 1) {
                // Case 1: 1 channel
                dst.at(newY, newX) = src.at(y, x);
            } else if (src.shape()[2] == 3) {
                // Case 2: 3 channels
                dst.at(newY, newX, 0) = src.at(y, x, 0);  // Blue
                dst.at(newY, newX, 1) = src.at(y, x, 1);  // Green
                dst.at(newY, newX, 2) = src.at(y, x, 2);  // Red
            } else {
                throw std::invalid_argument("Unsupported number of channels for flipping");
            }
        }
    }
}

}  // namespace processing
