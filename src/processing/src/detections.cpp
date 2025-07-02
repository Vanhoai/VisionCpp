//
// File        : detections.cpp
// Author      : Hinsun
// Date        : 2025-06-26
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#include "processing/detections.hpp"

#include "core/core.hpp"
#include "processing/processing.hpp"
#include "processing/transformations.hpp"

namespace processing {

    std::vector<Keypoint> HarrisCornerDetector::detectCorners(const core::TensorF32 &src) {
        if (src.empty())
            throw std::invalid_argument("Please provide a tensor with valid data");

        if (src.dimensions() != 3 && src.dimensions() != 2)
            throw std::invalid_argument("Harris corner detector requires a 2D or 3D tensor");

        if (src.dimensions() == 3 && src.shape()[2] != 3)
            throw std::invalid_argument(
                "Harris corner detector requires a 2D tensor or a 3D tensor with 3 channels (RGB)");

        core::TensorF32 grayscale;

        // Step 1: convert to grayscale if necessary
        if (src.dimensions() == 3)
            grayscale = Transformations::convertToGrayScale(src);
        else
            grayscale = src;

        // Step 2: Calculate spatial derivatives using Sobel operators
        const size_t height = grayscale.shape()[0];
        const size_t width = grayscale.shape()[1];

        core::TensorF32 Ixx({height, width}, 0.0f);
        core::TensorF32 Iyy({height, width}, 0.0f);
        core::TensorF32 Ixy({height, width}, 0.0f);

        for (size_t y = BORDER; y < height - BORDER; y++) {
            for (size_t x = BORDER; x < width - BORDER; x++) {
                float Ix = (grayscale.at(y, x + 1) - grayscale.at(y, x - 1)) * 0.5f;
                float Iy = (grayscale.at(y + 1, x) - grayscale.at(y - 1, x)) * 0.5f;

                Ixx.at(y, x) = Ix * Ix;
                Iyy.at(y, x) = Iy * Iy;
                Ixy.at(y, x) = Ix * Iy;
            }
        }

        // Step 3: Construct the structure tensor M(x, y) - Harris response
        core::TensorF32 R({height, width}, 0.0f);
        int khalf = WINDOW_SIZE / 2;

        for (size_t y = khalf; y < height - khalf; y++) {
            for (size_t x = khalf; x < width - khalf; x++) {
                float sumIxx = 0.0f, sumIyy = 0.0f, sumIxy = 0.0f;

                for (int dy = -khalf; dy <= khalf; ++dy) {
                    for (int dx = -khalf; dx <= khalf; ++dx) {
                        sumIxx += Ixx.at(y + dy, x + dx);
                        sumIyy += Iyy.at(y + dy, x + dx);
                        sumIxy += Ixy.at(y + dy, x + dx);
                    }
                }

                float det = sumIxx * sumIyy - sumIxy * sumIxy;
                float trace = sumIxx + sumIyy;
                R.at(y, x) = det - SENSITIVITY_FACTOR * trace * trace;
            }
        }

        // Step 4: Non-maximum suppression
        std::vector<Keypoint> keypoints;
        for (size_t y = khalf; y < height - khalf; y++) {
            for (size_t x = khalf; x < width - khalf; x++) {
                float val = R.at(y, x);
                if (val < THRESHOLD)
                    continue;

                bool isLocalMax = true;
                for (int dy = -1; dy <= 1 && isLocalMax; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0)
                            continue;

                        if (R.at(y + dy, x + dx) >= val) {
                            isLocalMax = false;
                            break;
                        }
                    }
                }

                if (isLocalMax) {
                    Keypoint kp;
                    kp.x = static_cast<float>(x);
                    kp.y = static_cast<float>(y);

                    kp.response = val;
                    keypoints.push_back(kp);
                }
            }
        }

        // Limit the number of keypoints to MAX_POINTS
        if (keypoints.size() > MAX_POINTS) {
            std::sort(keypoints.begin(), keypoints.end(),
                      [](const Keypoint &a, const Keypoint &b) { return a.response > b.response; });
            keypoints.resize(MAX_POINTS);
        }

        return keypoints;
    }

}   // namespace processing
