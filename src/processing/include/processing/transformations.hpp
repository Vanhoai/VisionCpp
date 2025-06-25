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

#include <core/core.hpp>
#include <core/tensor.hpp>

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
             * @param src           Input image (source).
             * @param dst           Output image (converted).
             * @param colorSpace   Target color space.
             */
            static void convertColorSpace(const core::Tensor<core::float32> &src,
                                          core::Tensor<core::float32> &dst, ColorSpace colorSpace);

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
             * @param src   Input image in BGR format.
             * @param dst   Output image in HSV format (3-channel float/double or scaled uchar).
             */
            static void convertBGRToHSV(const core::Tensor<core::float32> &src,
                                        core::Tensor<core::float32> &dst);

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
             * @param src: The input image in HSV format (3 channels)
             * @param dst: The output image in BGR format like opencv saved
             */
            static void convertHSVtoBGR(const core::Tensor<core::float32> &src,
                                        core::Tensor<core::float32> &dst);

            /**
             * @brief Convert a BGR image to grayscale.
             *
             * Formula used:
             *     Gray = 0.299 * R + 0.587 * G + 0.114 * B
             *
             * @param src   Input image in BGR format.
             * @param dst   Output grayscale image (1 channel).
             */
            static void convertToGrayScale(const core::Tensor<core::float32> &src,
                                           core::Tensor<core::float32> &dst);

            /**
             * @brief Resize an image to a new size.
             *
             * @param src       Input image.
             * @param dst       Output resized image.
             * @param width     New width of the image.
             * @param height    New height of the image.
             */
            static void resize(const core::Tensor<core::float32> &src,
                               core::Tensor<core::float32> &dst, size_t width, size_t height);

            /**
             * @brief Normalize an image by mean and standard deviation.
             *
             * For each channel: (value - mean) / std.
             *
             * @param src   Input image.
             * @param dst   Output normalized image.
             * @param mean  Mean value per channel.
             * @param std   Standard deviation per channel.
             */
            static void normalize(const core::Tensor<core::float32> &src,
                                  core::Tensor<core::float32> &dst,
                                  const std::vector<core::float32> &mean,
                                  const std::vector<core::float32> &std);

            /**
             * @brief Pad an image with a constant border.
             *
             * @param src           Input image.
             * @param dst           Output padded image.
             * @param padding       Size of padding applied to each side of the image.
             * @param value         Padding value (default: black).
             */
            static void pad(const core::Tensor<core::float32> &src,
                            core::Tensor<core::float32> &dst, core::float32 padding,
                            core::float32 value);

            /**
             * @brief Crop an image to a given region of interest (ROI).
             *
             * @param src Input image.
             * @param dst Output cropped image.
             * @param roi Region of interest to crop.
             */
            static void crop(const core::Tensor<core::float32> &src,
                             core::Tensor<core::float32> &dst, const core::Rect &roi);

            /**
             * @brief Randomly crop an image to a given size.
             *
             * @param src       Input image.
             * @param dst       Output cropped image.
             * @param width     Width of the crop.
             * @param height    Height of the crop.
             */
            static void randomCrop(const core::Tensor<core::float32> &src,
                                   core::Tensor<core::float32> &dst, size_t width, size_t height);

            /**
             * @brief Rotate an image by a specified angle.
             *
             * Supported angles:
             * - 90, 180, 270 degrees (clockwise)
             *
             * @param src   Input image.
             * @param dst   Rotated output image.
             * @param angle Rotation angle (clockwise).
             */
            static void rotate(const core::Tensor<core::float32> &src,
                               core::Tensor<core::float32> &dst, RotateAngle angle);

            /**
             * @brief Flip an image horizontally, vertically, or both.
             *
             * @param src       Input image.
             * @param dst       Flipped output image.
             * @param flipCode  Direction to flip:
             *                  - FlipCode::VERTICAL: Vertical flip (up-down).
             *                  - FlipCode::HORIZONTAL: Horizontal flip (left-right).
             *                  - FlipCode::BOTH: Both horizontal and vertical flip.
             */
            static void flip(const core::Tensor<core::float32> &src,
                             core::Tensor<core::float32> &dst, FlipCode flipCode);
    };

}   // namespace processing

#endif   // TRANSFORMATIONS_HPP
