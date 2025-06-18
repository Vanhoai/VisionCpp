//
// Created by Hinsun on 2025-06-18
// Copyright (c) 2025 VanHoai. All rights reserved.
//

#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include <opencv2/opencv.hpp>

namespace preprocessing {

    class Preprocessor {
        private:
        public:
            /**
             * @author Hinsun
             * @data 2025-06-18
             * @params:
             * inputImage: The input image in BGR format like opencv saved
             * outputImage: The output image in grayscale format (1 channel)
             * @brief Converts a BGR image to grayscale using the formula:
             * F(R, G, B) = 0.299 * R + 0.587 * G + 0.114 * B
             */
            static void convertToGrayScale(const cv::Mat &inputImage, cv::Mat &outputImage);
    };

}   // namespace preprocessing

#endif   // PREPROCESSING_HPP
