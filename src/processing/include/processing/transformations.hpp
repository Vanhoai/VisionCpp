//
// Created by Hinsun on 2025-06-18
// Copyright (c) 2025 VanHoai. All rights reserved.
//

#ifndef TRANSFORMATIONS_HPP
#define TRANSFORMATIONS_HPP

#include <opencv2/opencv.hpp>

namespace processing {

    /**
     * Transformations class provides a collection of static methods for image processing.
     * This class provides methods for converting images with many different including:
     * - Transform Color Space: BGR to HSV, HSV to BGR
     * - Convert to Grayscale
     * - Resize, Normalize
     * - Padding, Cropping, Random Cropping
     * - Rotate, Flip
     */
    class Transformations {
        public:
            enum class ColorSpace { BGR_TO_HSV, HSV_TO_BGR };
            enum class FlipCode { VERTICAL = 0, HORIZONTAL = 1, BOTH = -1 };
            enum class RotateAngle { CLOCKWISE_90 = 0, CLOCKWISE_180 = 1, CLOCKWISE_270 = 2 };

            /**
             * Convert an image to a specified color space.
             * This function takes an input image and converts it to the specified color space.
             * @params:
             * inputImage: The input image to be converted
             * outputImage: The converted output image
             * colorSpace: The desired color space for the output image
             *
             * Notice:
             * - ColorSpace::BGR_TO_HSV: Convert BGR to HSV
             * - ColorSpace::HSV_TO_BGR: Convert HSV to BGR
             */
            static void convertColorSpace(const cv::Mat &inputImage, cv::Mat &outputImage,
                                          ColorSpace colorSpace);

            /**
             * Convert a BGR image to HSV.
             * Function takes a BGR image as input and converts it to an HSV image.
             * @params:
             * inputImage: The input image in BGR format like opencv saved
             * outputImage: The output image in HSV format (3 channels)
             *
             * Notice:
             * HSV stands for Hue, Saturation, and Value.
             * Hue represents the color type, Saturation represents the intensity of the color,
             * and Value represents the brightness of the color.
             * For convert BGR to HSV, follow formula:
             * 1. Calculate the normalized RGB values, by divided by 255 to change the range
             * from [0...255] to [0...1]:
             * - R' = R / 255.0
             * - G' = G / 255.0
             * - B' = B / 255.0
             *
             * 2. Calculate the maximum and minimum values of R', G', B':
             * - Cmax = max(R', G', B')
             * - Cmin = min(R', G', B')
             *
             * 3. Calculate delta (Δ) - the difference between Cmax and Cmin:
             * - Δ = Cmax - Cmin
             *
             * 4. Calculate the Hue (H) with 4 cases:
             * - Δ = 0, then H = 0 (undefined hue)
             * - Cmax = R', then H = 60 * ((G' - B') / Δ) % 360
             * - Cmax = G', then H = 60 * ((B' - R') / Δ + 2) % 360
             * - Cmax = B', then H = 60 * ((R' - G') / Δ + 4) % 360
             *
             * 5. Calculate the Saturation (S) with 2 cases:
             * - Cmax = 0, then S = 0 (undefined saturation)
             * - S = Δ / Cmax
             *
             * 6. Calculate the Value (V): V = Cmax
             *
             * Notice: cv::Mat in OpenCV is stored in BGR format.
             */
            static void convertBGRToHSV(const cv::Mat &inputImage, cv::Mat &outputImage);

            /**
             * Convert an HSV image to BGR.
             * Function takes an HSV image as input and converts it to a BGR image.
             * @params:
             * inputImage: The input image in HSV format (3 channels)
             * outputImage: The output image in BGR format like opencv saved
             *
             * Notice:
             * For convert HSV to BGR, follow formula:
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
            static void convertHSVtoBGR(const cv::Mat &inputImage, cv::Mat &outputImage);

            /**
             * Convert a BGR image to grayscale.
             * This function takes a BGR image as input and converts it to a grayscale image using
             * the formula: F(R, G, B) = 0.299 * R + 0.587 * G + 0.114 * B
             * @params:
             * inputImage: The input image in BGR format like opencv saved
             * outputImage: The output image in grayscale format (1 channel)
             */
            static void convertToGrayScale(const cv::Mat &inputImage, cv::Mat &outputImage);

            /**
             * Resize an image to a new size.
             * This function takes an input image and resizes it to the specified new size.
             * @params:
             * inputImage: The input image to be resized
             * outputImage: The resized output image
             * newSize: The desired size for the output image
             */
            static void resize(const cv::Mat &inputImage, cv::Mat &outputImage,
                               const cv::Size &newSize);

            /**
             * Normalize an image with mean and standard deviation.
             * This function takes an input image and normalizes it by subtracting the mean
             * and dividing by the standard deviation.
             * @param:
             * inputImage: The input image to be normalized
             * outputImage: The normalized output image
             * mean: The mean value for normalization (default is (0, 0, 0))
             * std: The standard deviation value for normalization (default is (1, 1, 1))
             */
            static void normalize(const cv::Mat &inputImage, cv::Mat &outputImage,
                                  const cv::Scalar &mean, const cv::Scalar &std);

            /**
             * Pad an image with a specified padding size and value.
             * This function takes an input image and pads it with the specified size and value.
             * @params:
             * inputImage: The input image to be padded
             * outputImage: The padded output image
             * paddingSize: The size of the padding to be applied (width, height)
             * value: The value used for padding (default is black color)
             */
            static void pad(const cv::Mat &inputImage, cv::Mat &outputImage,
                            const cv::Size &paddingSize, const cv::Scalar &value);

            /**
             * Crop an image to a specified region of interest (ROI).
             * This function takes an input image and crops it to the specified ROI.
             * @param:
             * inputImage: The input image to be cropped
             * outputImage: The cropped output image
             * roi: The region of interest (ROI) to crop the image
             * Notice: The roi must be within the bounds of the input image.
             * If the roi is out of bounds, it will throw an exception.
             */
            static void crop(const cv::Mat &inputImage, cv::Mat &outputImage, const cv::Rect &roi);

            /**
             * Randomly crop an image to a specified size.
             * This function takes an input image and randomly crops it to the specified size.
             * @params:
             * inputImage: The input image to be cropped
             * outputImage: The randomly cropped output image
             * cropSize: The desired size for the cropped image
             * Notice: If the cropSize is larger than the input image size, it will throw an
             * exception.
             */
            static void randomCrop(const cv::Mat &inputImage, cv::Mat &outputImage,
                                   const cv::Size &cropSize);

            /**
             * Rotate an image by a specified angle.
             * This function takes an input image and rotates it by the specified angle.
             * @params:
             * inputImage: The input image to be rotated
             * outputImage: The rotated output image
             * angle: The angle in degrees to rotate the image
             */
            static void rotate(const cv::Mat &inputImage, cv::Mat &outputImage, RotateAngle angle);

            /**
             * Flip an image horizontally or vertically.
             * This function takes an input image and flips it either horizontally or vertically.
             * @params:
             * inputImage: The input image to be flipped
             * outputImage: The flipped output image
             * flipCode: The code to specify the flip direction (0 for vertical, 1 for horizontal,
             * -1 for both)
             */
            static void flip(const cv::Mat &inputImage, cv::Mat &outputImage, FlipCode flipCode);
    };

}   // namespace processing

#endif   // TRANSFORMATIONS_HPP
