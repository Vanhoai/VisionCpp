//
// File        : transformations.cpp
// Author      : Hinsun
// Date        : 2025-06-TODAY
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#ifndef TRANSFORMATIONS_HPP
#define TRANSFORMATIONS_HPP

/**
 * @brief Header for image transformation utilities.
 *
 * This file declares the `processing::Transformations` class, which provides
 * static methods for common image preprocessing tasks such as:
 * - Color space conversions (BGR <-> HSV)
 * - Grayscale conversion
 * - Image resizing and normalization
 * - Padding and cropping (including random cropping)
 * - Rotation and flipping
 */

#include <opencv2/opencv.hpp>

namespace processing {

    class Transformations {
        public:
            /**
             * @enum ColorSpace
             * @brief Enum to specify color space conversion types.
             */
            enum class ColorSpace { BGR_TO_HSV, HSV_TO_BGR };

            /**
             * @enum FlipCode
             * @brief Enum to specify flip directions.
             */
            enum class FlipCode { VERTICAL = 0, HORIZONTAL = 1, BOTH = -1 };

            /**
             * @enum RotateAngle
             * @brief Enum to specify rotation angles (clockwise).
             */
            enum class RotateAngle { CLOCKWISE_90 = 0, CLOCKWISE_180 = 1, CLOCKWISE_270 = 2 };

            /**
             * @brief Convert an image to a specified color space.
             *
             * Supported conversions:
             * - ColorSpace::BGR_TO_HSV — Convert from BGR to HSV.
             * - ColorSpace::HSV_TO_BGR — Convert from HSV to BGR.
             *
             * @param inputImage   Input image (source).
             * @param outputImage  Output image (converted).
             * @param colorSpace   Target color space.
             */
            static void convertColorSpace(const cv::Mat &inputImage, cv::Mat &outputImage,
                                          ColorSpace colorSpace);

            /**
             * @brief Convert a BGR image to HSV format.
             *
             * This function takes an image in BGR format and converts it to HSV format.
             * HSV stands for Hue, Saturation, and Value:
             * - Hue represents the type of color (e.g., red, blue).
             * - Saturation represents the vibrancy of the color.
             * - Value represents the brightness of the color.
             *
             * Conversion steps:
             * 1. Normalize the BGR values to the range [0, 1]:
             *    - R' = R / 255.0
             *    - G' = G / 255.0
             *    - B' = B / 255.0
             *
             * 2. Compute:
             *    - Cmax = max(R', G', B')
             *    - Cmin = min(R', G', B')
             *    - Δ = Cmax - Cmin
             *
             * 3. Compute Hue (H):
             *    - If Δ == 0 → H = 0
             *    - If Cmax == R' → H = 60 * fmod((G' - B') / Δ, 6)
             *    - If Cmax == G' → H = 60 * (((B' - R') / Δ) + 2)
             *    - If Cmax == B' → H = 60 * (((R' - G') / Δ) + 4)
             *    - H should be in [0, 360)
             *
             * 4. Compute Saturation (S):
             *    - If Cmax == 0 → S = 0
             *    - Else → S = Δ / Cmax
             *
             * 5. Compute Value (V): V = Cmax
             *
             * Notes:
             * - Input image must be in BGR format (as used by OpenCV).
             * - Output image will have 3 channels: H in [0, 360), S and V in [0, 1] (or scaled
             * appropriately).
             *
             * @param inputImage  Input image in BGR format.
             * @param outputImage Output image in HSV format (3-channel float/double or scaled
             * uchar).
             */
            static void convertBGRToHSV(const cv::Mat &inputImage, cv::Mat &outputImage);

            /**
             * @brief Convert an HSV image to BGR format.
             *
             * This function takes an HSV image as input and converts it to a BGR image.
             *
             * Conditions required:
             * - Hue (H) must be in the range [0, 360]
             * - Saturation (S) must be in the range [0, 1]
             * - Value (V) must be in the range [0, 1]
             *
             * Conversion steps:
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
             *
             * @param inputImage: The input image in HSV format (3 channels)
             * @param outputImage: The output image in BGR format like opencv saved
             */
            static void convertHSVtoBGR(const cv::Mat &inputImage, cv::Mat &outputImage);

            /**
             * @brief Convert a BGR image to grayscale.
             *
             * Formula used:
             *     Gray = 0.299 * R + 0.587 * G + 0.114 * B
             *
             * @param inputImage   Input image in BGR format.
             * @param outputImage  Output grayscale image (1 channel).
             */
            static void convertToGrayScale(const cv::Mat &inputImage, cv::Mat &outputImage);

            /**
             * @brief Resize an image to a new size.
             *
             * @param inputImage   Input image.
             * @param outputImage  Output resized image.
             * @param newSize      Target size (width, height).
             */
            static void resize(const cv::Mat &inputImage, cv::Mat &outputImage,
                               const cv::Size &newSize);

            /**
             * @brief Normalize an image by mean and standard deviation.
             *
             * For each channel: (value - mean) / std.
             *
             * @param inputImage   Input image.
             * @param outputImage  Output normalized image.
             * @param mean         Mean value per channel.
             * @param std          Standard deviation per channel.
             */
            static void normalize(const cv::Mat &inputImage, cv::Mat &outputImage,
                                  const cv::Scalar &mean, const cv::Scalar &std);

            /**
             * @brief Pad an image with a constant border.
             *
             * @param inputImage    Input image.
             * @param outputImage   Output padded image.
             * @param paddingSize   Size of padding (left/right, top/bottom).
             * @param value         Padding value (default: black).
             */
            static void pad(const cv::Mat &inputImage, cv::Mat &outputImage,
                            const cv::Size &paddingSize, const cv::Scalar &value);

            /**
             * @brief Crop an image to a given region of interest (ROI).
             *
             * @param inputImage   Input image.
             * @param outputImage  Output cropped image.
             * @param roi          Region of interest to crop.
             *
             * @throw std::out_of_range If the ROI is outside the image bounds.
             */
            static void crop(const cv::Mat &inputImage, cv::Mat &outputImage, const cv::Rect &roi);

            /**
             * @brief Randomly crop an image to a given size.
             *
             * @param inputImage   Input image.
             * @param outputImage  Output cropped image.
             * @param cropSize     Target crop size (width, height).
             *
             * @throw std::invalid_argument If cropSize is larger than the image size.
             */
            static void randomCrop(const cv::Mat &inputImage, cv::Mat &outputImage,
                                   const cv::Size &cropSize);

            /**
             * @brief Rotate an image by a specified angle.
             *
             * Supported angles:
             * - 90, 180, 270 degrees (clockwise)
             *
             * @param inputImage   Input image.
             * @param outputImage  Rotated output image.
             * @param angle        Rotation angle (clockwise).
             */
            static void rotate(const cv::Mat &inputImage, cv::Mat &outputImage, RotateAngle angle);

            /**
             * @brief Flip an image horizontally, vertically, or both.
             *
             * @param inputImage   Input image.
             * @param outputImage  Flipped output image.
             * @param flipCode     Direction to flip:
             *                     - FlipCode::VERTICAL: Vertical flip (up-down).
             *                     - FlipCode::HORIZONTAL: Horizontal flip (left-right).
             *                     - FlipCode::BOTH: Both horizontal and vertical flip.
             */
            static void flip(const cv::Mat &inputImage, cv::Mat &outputImage, FlipCode flipCode);
    };

}   // namespace processing

#endif   // TRANSFORMATIONS_HPP
