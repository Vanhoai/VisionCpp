//
// Created by VanHoai on 30/5/25.
//

#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <opencv2/opencv.hpp>

class ImageProcessing {
    private:
        cv::Mat original;
        cv::Mat processed;

    public:
        explicit ImageProcessing(cv::Mat &originalImage)
            : original(std::move(originalImage)) {}

        [[nodiscard]] cv::Mat getOriginal() const { return original; }
        [[nodiscard]] cv::Mat getProcessed() const { return processed; }

        void convertToGrayScale();
};

#endif   // IMAGE_PROCESSING_H
