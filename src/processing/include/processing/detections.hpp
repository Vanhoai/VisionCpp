//
// File        : detections.hpp
// Author      : Hinsun
// Date        : 2025-06-26
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#ifndef DETECTIONS_HPP
#define DETECTIONS_HPP

/**
 * @brief Header for object detection operations.
 *
 * This file declares classes, which provides methods for various object detection algorithms.
 * - Harris Corner Detection.
 * - Haar Cascade Classifier
 * - HOG (Histogram of Oriented Gradients)
 * - Contour Detection
 */

#include <vector>

#include "core/core.hpp"
#include "processing/processing.hpp"

namespace processing {

    /**
     * @brief Class for performing Harris Corner Detection.
     *
     * This class provides methods to detect corners in images using the Harris corner detection
     * algorithm.
     *
     * How HCD works:
     * 1. Color to grayscale conversion
     *    If we use Harris corner detector in a color image, we need to convert it to grayscale,
     *    which enhance the processing speed and reduce the complexity.
     * 2. Spatial derivative calculation
     *    Next, we are going to find the spatial derivatives of the image in both x and y
     *    directions. This can be approximated by applying Sobel operators.
     * 3. Structure tensor setup
     *    With Ix(x, y), Iy(x, y), we can construct the structure tensor M(x, y) as follows:
     *    M(x, y) = [Ix^2(x, y)  Ix(x, y) * Iy(x, y)]
     * 4. Harris response calculation
     *    The Harris response R(x, y) is calculated using the determinant and trace of the
     *    structure tensor:
     *    R(x, y) = det(M(x, y)) - k * (trace(M(x, y)))^2
     *    where k is a sensitivity factor (usually between 0.04 and 0.06).
     * 5. Non-maximum suppression
     *    To find the corners, we apply non-maximum suppression to the Harris response map.
     */
    class HarrisCornerDetector {
        private:
            static constexpr float SENSITIVITY_FACTOR = 0.04f;   // Sensitivity factor
            static constexpr float THRESHOLD = 0.01f;            // Threshold for corner response
            static constexpr int WINDOW_SIZE = 3;                // Size for summing gradients
            static constexpr int BORDER = 1;                     // Padding to avoid edge issues
            static constexpr int MAX_POINTS = 1000;              // Maximum number of corners

        public:
            /**
             * @brief Detect corners in an image using Harris corner detection.
             *
             * @param src input image tensor
             * @return vector of detected keypoints
             */
            static std::vector<Keypoint> detectCorners(const core::TensorF32 &src);
    };

}   // namespace processing

#endif   // DETECTIONS_HPP
