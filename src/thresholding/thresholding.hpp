#ifndef THRESHOLDING_H
#define THRESHOLDING_H

#include <opencv2/opencv.hpp>

#include "../core/base/image_processing/image_processing.hpp"

/**
 * Thresholding class for image processing.
 * This class will be implemented to perform various thresholding techniques
 *
 * 1. Global Thresholding
 * 2. Binary Thresholding
 * 3. Inverse Binary Thresholding
 * 4. Truncated Thresholding
 * 5. To Zero Thresholding
 * 6. Inverse To Zero Thresholding
 */

enum class ThresholdingType {
    THRESH_BINARY,
    THRESH_BINARY_INV,
    THRESH_TRUNC,
    THRESH_TOZERO,
    THRESH_TOZERO_INV
};

class Thresholding : public ImageProcessing {
    public:
        explicit Thresholding(cv::Mat &image) : ImageProcessing(image) {
            convertToGrayScale();
        }

        /**
         * Get the Otsu's threshold value for the image.
         * These algorithms are used to find the optimal threshold value by
         * divided into two classes, foreground and background.
         *
         * Following steps:
         * 1.Calculate histogram of the image
         * 2.Each value in the histogram do:
         *     + Wo: sum of probabilities group 1 (<= t)
         *     + W1: sum of probabilities group 2 (> t)
         *     + Uo: mean of group 1 (<= t)
         *     + U1: mean of group 2 (> t)
         *     + UT: mean of the whole image
         *     + Ob^2: value need optimized
         */
        [[nodiscard]] int otsuThresholding() const;
        [[nodiscard]] cv::Mat applyThresholding(int thresh, int maxVal,
                                                ThresholdingType type) const;

        [[nodiscard]] int openCVOtsu(int thresh, int maxVal) const;
        void compareWithOpencv(int thresh, int maxVal) const;
};

#endif   // THRESHOLDING_H
