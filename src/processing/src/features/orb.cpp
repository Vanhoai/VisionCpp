//
// File        : orb.cpp
// Author      : Hinsun
// Date        : 2025-06-TODAY
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#include <cmath>
#include <random>

#include "core/core.hpp"
#include "processing/features.hpp"
#include "processing/operators.hpp"
#include "processing/transformations.hpp"

namespace processing {

Layers ORB::buildImagePyramid(const core::Tensor<core::float32> &src) {
    Layers pyramid(ORB::NUM_LEVELS);

    core::Tensor<core::float32> grayscale = Transformations::convertToGrayScale(src);
    pyramid[0].resize(1);
    pyramid[0][0] = grayscale;

    for (int level = 1; level < ORB::NUM_LEVELS; level++) {
        pyramid[level].resize(1);
        pyramid[level][0] = processing::downsample(pyramid[level - 1][0]);
    }

    return pyramid;
}

bool ORB::isFASTCorner(const core::TensorF32 &src, int x, int y, float threshold, int n) {
    const int height = src.height();
    const int width = src.width();

    // avoid out of bounds access
    if (x < 3 || x >= width - 3 || y < 3 || y >= height - 3) return false;

    // FAST-9 circle pattern (16 pixels)
    const int circle[16][2] = {
        {0,  -3},
        {1,  -3},
        {2,  -2},
        {3,  -1},
        {3,  0 },
        {3,  1 },
        {2,  2 },
        {1,  3 },
        {0,  3 },
        {-1, 3 },
        {-2, 2 },
        {-3, 1 },
        {-3, 0 },
        {-3, -1},
        {-2, -2},
        {-1, -3}
    };

    // Check if n contiguous pixels are all brighter or all darker
    int brightCount = 0, darkCount = 0;
    int maxBrightSeq = 0, maxDarkSeq = 0;
    int currentBrightSeq = 0, currentDarkSeq = 0;

    const float centerPixel = src.at(y, x);
    for (auto &point : circle) {
        const float pixel = src.at(y + point[1], x + point[0]);
        const float diff = pixel - centerPixel;

        if (diff > threshold) {
            brightCount++;
            currentBrightSeq++;
            currentDarkSeq = 0;
            maxBrightSeq = std::max(maxBrightSeq, currentBrightSeq);
        } else if (diff < -threshold) {
            darkCount++;
            currentDarkSeq++;
            currentBrightSeq = 0;
            maxDarkSeq = std::max(maxDarkSeq, currentDarkSeq);
        } else {
            currentBrightSeq = 0;
            currentDarkSeq = 0;
        }
    }

    // Check wraparound for contiguous sequences
    if (brightCount >= n) {
        int wraparoundBright = 0;

        for (int i = 0; i < n - 1; i++) {
            const float pixel = src.at(y + circle[i][1], x + circle[i][0]);
            if (pixel - centerPixel > threshold)
                wraparoundBright++;
            else
                break;
        }

        for (int i = 15; i >= 16 - (n - 1 - wraparoundBright); i--) {
            const float pixel = src.at(y + circle[i][1], x + circle[i][0]);
            if (pixel - centerPixel > threshold)
                wraparoundBright++;
            else
                break;
        }

        if (maxBrightSeq >= n || wraparoundBright >= n) return true;
    }

    if (darkCount >= n) {
        int wraparoundDark = 0;

        for (int i = 0; i < n - 1; i++) {
            const float pixel = src.at(y + circle[i][1], x + circle[i][0]);
            if (centerPixel - pixel > threshold)
                wraparoundDark++;
            else
                break;
        }

        for (int i = 15; i >= 16 - (n - 1 - wraparoundDark); i--) {
            const float pixel = src.at(y + circle[i][1], x + circle[i][0]);
            if (centerPixel - pixel > threshold)
                wraparoundDark++;
            else
                break;
        }

        if (maxDarkSeq >= n || wraparoundDark >= n) return true;
    }

    return false;
}

float ORB::computeCentroidAngle(const core::Tensor<core::float32> &src, int x, int y, int radius) {
    const int height = src.shape()[0];
    const int width = src.shape()[1];

    float m01 = 0, m10 = 0;

    for (int v = -radius; v <= radius; v++) {
        for (int u = -radius; u <= radius; u++) {
            if (u * u + v * v > radius * radius) continue;

            const int px = x + u;
            const int py = y + v;

            if (px >= 0 && px < width && py >= 0 && py < height) {
                const float intensity = src.at(py, px);
                m01 += v * intensity;
                m10 += u * intensity;
            }
        }
    }

    return std::atan2(m01, m10);
}

float ORB::computeHarrisScore(const core::TensorF32 &src, int x, int y) {
    const int height = src.height();
    const int width = src.width();

    // x, y must inside the image bounds
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return 0.0f;

    // Compute gradients using Sobel operators
    float Ixx = 0.0f, Iyy = 0.0f, Ixy = 0.0f;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            const int px = x + dx;
            const int py = y + dy;

            if (py < 0 || py >= height || px < 0 || px >= width) continue;

            const float Ix = (src.at(y + dy, x + dx + 1) - src.at(y + dy, x + dx - 1)) * 0.5f;
            const float Iy = (src.at(y + dy + 1, x + dx) - src.at(y + dy - 1, x + dx)) * 0.5f;

            Ixx += Ix * Ix;
            Iyy += Iy * Iy;
            Ixy += Ix * Iy;
        }
    }

    // Compute Harris response
    const float det = Ixx * Iyy - Ixy * Ixy;
    const float trace = Ixx + Iyy;
    constexpr float k = 0.04f;
    return det - k * trace * trace;
}

std::vector<Keypoint> ORB::detectFASTKeypoints(const Layers &pyramid) {
    std::vector<Keypoint> keypoints;

    for (int level = 0; level < NUM_LEVELS; level++) {
        // FIXME: change pyramid type from Layers to std::vector<core::TensorF32>
        const auto &image = pyramid[level][0];

        const int height = image.shape()[0];
        const int width = image.shape()[1];
        const float scale = std::pow(SCALE_FACTOR, level);

        std::vector<Keypoint> levelKeypoints;

        // FAST corner detection
        for (int y = BORDER_SIZE; y < height - BORDER_SIZE; y++) {
            for (int x = BORDER_SIZE; x < width - BORDER_SIZE; x++) {
                if (isFASTCorner(image, x, y, FAST_THRESHOLD, 9)) {
                    Keypoint kp;
                    kp.x = x * scale;
                    kp.y = y * scale;
                    kp.scale = scale;
                    kp.octave = level;
                    kp.layer = 0;
                    kp.response = computeHarrisScore(image, x, y);
                    levelKeypoints.push_back(kp);
                }
            }
        }

        // Sort by Harris score and keep top keypoints
        std::sort(levelKeypoints.begin(), levelKeypoints.end(),
                  [](const Keypoint &a, const Keypoint &b) { return a.response > b.response; });

        constexpr int maxKeypointsPerLevel = MAX_KEYPOINTS / NUM_LEVELS;
        if (levelKeypoints.size() > maxKeypointsPerLevel)
            levelKeypoints.resize(maxKeypointsPerLevel);

        keypoints.insert(keypoints.end(), levelKeypoints.begin(), levelKeypoints.end());
    }

    return keypoints;
}

void ORB::computeOrientations(std::vector<Keypoint> &keypoints, const Layers &pyramid) {
    for (auto &kp : keypoints) {
        const auto &image = pyramid[kp.octave][0];
        const int x = static_cast<int>(kp.x / kp.scale);
        const int y = static_cast<int>(kp.y / kp.scale);
        kp.angle = computeCentroidAngle(image, x, y, PATCH_SIZE / 2);
    }
}

std::vector<uint8_t> ORB::computeBRIEFDescriptor(const core::TensorF32 &src, const Keypoint &kp) {
    const int x = static_cast<int>(kp.x);
    const int y = static_cast<int>(kp.y);
    const float angle = kp.angle;

    std::vector<uint8_t> descriptor(32, 0);  // 256 bits = 32 bytes

    const float cosAngle = std::cos(angle);
    const float sinAngle = std::sin(angle);

    // Rotate the BRIEF pattern based on keypoint orientation
    for (int i = 0; i < 256; i++) {
        const auto &pair = BRIEF_PATTERN[i];

        // Rotate first point
        const float x1_rot = pair.first.first * cosAngle - pair.first.second * sinAngle;
        const float y1_rot = pair.first.first * sinAngle + pair.first.second * cosAngle;

        // Rotate second point
        const float x2_rot = pair.second.first * cosAngle - pair.second.second * sinAngle;
        const float y2_rot = pair.second.first * sinAngle + pair.second.second * cosAngle;

        const int px1 = x + static_cast<int>(x1_rot);
        const int py1 = y + static_cast<int>(y1_rot);
        const int px2 = x + static_cast<int>(x2_rot);
        const int py2 = y + static_cast<int>(y2_rot);

        // Check bounds
        const int height = src.shape()[0];
        const int width = src.shape()[1];

        if (px1 >= 0 && px1 < width && py1 >= 0 && py1 < height && px2 >= 0 && px2 < width &&
            py2 >= 0 && py2 < height) {
            if (src.at(py1, px1) < src.at(py2, px2)) {
                descriptor[i / 8] |= (1 << (i % 8));
            }
        }
    }

    return descriptor;
}

void ORB::computeDescriptors(std::vector<Keypoint> &keypoints, const Layers &pyramid) {
    for (auto &kp : keypoints) {
        const auto &image = pyramid[kp.octave][0];

        // Smooth the image slightly for better descriptor quality
        const auto smoothed = gaussianBlur(image, 2.0f);

        const auto briefDesc = computeBRIEFDescriptor(smoothed, kp);

        // Convert to float descriptor for consistency with SIFT interface
        kp.descriptor.resize(briefDesc.size());
        for (size_t i = 0; i < briefDesc.size(); i++) {
            kp.descriptor[i] = static_cast<float>(briefDesc[i]);
        }
    }
}

std::vector<Keypoint> ORB::detectAndCompute(const core::Tensor<core::float32> &src) {
    // Step 1: Build image pyramid
    double start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const Layers pyramid = buildImagePyramid(src);
    double end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::cout << "Build pyramid time: " << (end - start) / 1e6 << " ms" << std::endl;

    // Step 2: Detect FAST keypoints
    start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::vector<Keypoint> keypoints = detectFASTKeypoints(pyramid);
    end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::cout << "Detect FAST keypoints time: " << (end - start) / 1e6 << " ms" << std::endl;

    // Step 3: Compute orientations for keypoints
    start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    computeOrientations(keypoints, pyramid);
    end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::cout << "Compute orientations time: " << (end - start) / 1e6 << " ms" << std::endl;

    // Step 4: Compute BRIEF descriptors for keypoints
    start = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    computeDescriptors(keypoints, pyramid);
    end = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::cout << "Compute descriptors time: " << (end - start) / 1e6 << " ms" << std::endl;

    return keypoints;
}

std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> ORB::BRIEF_PATTERN =
    ORB::generateBRIEFPattern();

std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> ORB::generateBRIEFPattern() {
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> pattern;
    pattern.reserve(256);

    // Generate a fixed random pattern for reproducibility
    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<float> dist(0.0f, PATCH_SIZE / 5.0f);

    for (int i = 0; i < 256; i++) {
        const int x1 =
            static_cast<int>(std::clamp(dist(rng), -PATCH_SIZE / 2.0f, PATCH_SIZE / 2.0f));
        const int y1 =
            static_cast<int>(std::clamp(dist(rng), -PATCH_SIZE / 2.0f, PATCH_SIZE / 2.0f));
        const int x2 =
            static_cast<int>(std::clamp(dist(rng), -PATCH_SIZE / 2.0f, PATCH_SIZE / 2.0f));
        const int y2 =
            static_cast<int>(std::clamp(dist(rng), -PATCH_SIZE / 2.0f, PATCH_SIZE / 2.0f));

        pattern.emplace_back(std::make_pair(x1, y1), std::make_pair(x2, y2));
    }

    return pattern;
}

}  // namespace processing
