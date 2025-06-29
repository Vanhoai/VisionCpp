//
// File        : features.cpp
// Author      : Hinsun
// Date        : 2025-06-26
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#include "processing/features.hpp"

#include <iostream>

#include "core/core.hpp"
#include "processing/transformations.hpp"

namespace processing {

    core::Tensor<core::float32> SIFT::downsample(const core::Tensor<core::float32> &src) {
        const std::vector<size_t> &shape = src.shape();
        const size_t height = shape[0];
        const size_t width = shape[1];

        const size_t newHeight = height / 2;
        const size_t newWidth = width / 2;

        core::Tensor<core::float32> dst(newHeight, newWidth);

        for (size_t y = 0; y < newHeight; y++)
            for (size_t x = 0; x < newWidth; x++) dst.at(y, x) = src.at(y * 2, x * 2);

        return dst;
    }

    core::Tensor<core::float32> SIFT::convolveHorizontal(const core::Tensor<core::float32> &src,
                                                         const std::vector<float> &kernel) {
        const std::vector<size_t> &shape = src.shape();
        const size_t height = shape[0];
        const size_t width = shape[1];

        const int ksize = static_cast<int>(kernel.size());
        const int khalf = ksize / 2;

        core::Tensor<core::float32> dst(height, width);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                float sum = 0.0f;

                for (int k = -khalf; k <= khalf; ++k) {
                    int xx =
                        std::clamp<int>(static_cast<int>(x) + k, 0, static_cast<int>(width - 1));
                    sum += src(y, xx) * kernel[k + khalf];
                }

                dst(y, x) = sum;
            }
        }

        return dst;
    }

    core::Tensor<core::float32> SIFT::convolveVertical(const core::Tensor<core::float32> &src,
                                                       const std::vector<float> &kernel) {
        const std::vector<size_t> &shape = src.shape();
        const size_t height = shape[0];
        const size_t width = shape[1];

        const int ksize = static_cast<int>(kernel.size());
        const int khalf = ksize / 2;

        core::Tensor<core::float32> dst(height, width);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                float sum = 0.0f;

                for (int k = -khalf; k <= khalf; ++k) {
                    int yy =
                        std::clamp<int>(static_cast<int>(y) + k, 0, static_cast<int>(height - 1));
                    sum += src(yy, x) * kernel[k + khalf];
                }

                dst(y, x) = sum;
            }
        }

        return dst;
    }

    core::Tensor<core::float32> SIFT::gaussianBlur(const core::Tensor<core::float32> &src,
                                                   const float sigma) {
        int kernelSize = static_cast<int>(6 * sigma) | 1;   // Ensure odd size
        std::vector<float> kernel(kernelSize);

        float sum = 0.0f;
        int half = kernelSize / 2;

        for (int i = 0; i < kernelSize; i++) {
            float x = i - half;
            kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
            sum += kernel[i];
        }

        // Normalize kernel
        for (float &val : kernel) val /= sum;

        // Apply separable convolution (horizontal then vertical)
        core::Tensor<core::float32> temp = convolveHorizontal(src, kernel);
        return convolveVertical(temp, kernel);
    }

    core::Tensor<core::float32> SIFT::substract(const core::Tensor<core::float32> &a,
                                                const core::Tensor<core::float32> &b) {
        if (a.shape() != b.shape())
            throw std::invalid_argument("Tensors must have the same shape for subtraction.");

        core::Tensor<core::float32> dst(a.shape());
        for (size_t i = 0; i < a.size(); i++) dst.data()[i] = a.data()[i] - b.data()[i];
        return dst;
    }

    bool SIFT::isLocalExtremum(int x, int y, const core::Tensor<core::float32> &current,
                               const core::Tensor<core::float32> &below,
                               const core::Tensor<core::float32> &above) {
        float val = current.at(y, x);
        bool isMax = true, isMin = true;

        // Check 3x3x3 neighborhood
        for (int dz = -1; dz <= 1; dz++) {
            const auto &layer = (dz == -1) ? below : (dz == 0) ? current : above;

            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0 && dz == 0)
                        continue;

                    float neighbor = layer.at(y + dy, x + dx);
                    if (neighbor >= val)
                        isMax = false;

                    if (neighbor <= val)
                        isMin = false;

                    if (!isMax && !isMin)
                        return false;
                }
            }
        }

        return isMax || isMin;
    }

    Layers SIFT::buildScaleSpace(const core::Tensor<core::float32> &src) {
        // 4 octaves, 5 scales per octave (s+3 where s=2)
        // Example size of input: 512 x 512
        // Octave 0: [blur00, blur01, blur02, blur03, blur04] -> Size: 512 x 512
        // Octave 1: [blur10, blur11, blur12, blur13, blur14] -> Size: 256 x 256
        // Octave 2: [blur20, blur21, blur22, blur23, blur24] -> Size: 128 x 128
        // Octave 3: [blur30, blur31, blur32, blur33, blur34] -> Size: 64 x 64
        // Each blur is a Gaussian blur with increasing sigma

        Layers scaleSpace(NUM_OCTAVES);

        core::Tensor<core::float32> grayscale;
        processing::Transformations::convertToGrayScale(src, grayscale);

        for (int octave = 0; octave < NUM_OCTAVES; octave++) {
            scaleSpace[octave].resize(NUM_SCALES);

            // Get base image for this octave
            core::Tensor<core::float32> baseImage;
            if (octave == 0) {
                baseImage = grayscale;
            } else {
                // Downsample the previous octave's base image
                baseImage = downsample(scaleSpace[octave - 1][NUM_SCALES - 3]);
            }

            for (int scale = 0; scale < NUM_SCALES; scale++) {
                float sigma = SIGMA_0 * std::pow(K, scale);
                scaleSpace[octave][scale] = gaussianBlur(baseImage, sigma);
            }
        }

        return scaleSpace;
    }

    Layers SIFT::buildDoGSpace(const Layers &scaleSpace) {
        Layers dogSpace(NUM_OCTAVES);

        for (int octave = 0; octave < NUM_OCTAVES; octave++) {
            dogSpace[octave].resize(NUM_SCALES - 1);

            for (int scale = 0; scale < NUM_SCALES - 1; scale++)
                dogSpace[octave][scale] =
                    substract(scaleSpace[octave][scale + 1], scaleSpace[octave][scale]);
        }

        return dogSpace;
    }

    std::vector<Keypoint> SIFT::findKeypointCandidates(const Layers &dogSpace) {
        std::vector<Keypoint> candidates;

        for (int octave = 0; octave < NUM_OCTAVES; octave++) {
            // skip first and last scales
            for (int scale = 0; scale < NUM_SCALES - 2; scale++) {
                const auto &current = dogSpace[octave][scale];
                const auto &below = dogSpace[octave][scale - 1];
                const auto &above = dogSpace[octave][scale + 1];

                const size_t height = current.shape()[0];
                const size_t width = current.shape()[1];

                for (size_t y = 1; y < height - 1; y++) {
                    for (size_t x = 1; x < width - 1; x++) {
                        float val = current.at(y, x);
                        if (std::abs(val) < CONTRAST_THRESHOLD)
                            continue;

                        // Check if it's a local extremum
                        if (isLocalExtremum(x, y, current, below, above)) {
                            Keypoint kp;
                            kp.x = x * std::pow(2.0f, octave);
                            kp.y = y * std::pow(2.0f, octave);
                            kp.scale = SIGMA_0 * std::pow(K, scale) * std::pow(2.0f, octave);
                            kp.octave = octave;
                            kp.layer = scale;
                            candidates.push_back(kp);
                        }
                    }
                }
            }
        }

        return candidates;
    }

    // Implementation of SIFT feature extraction methods would go here
    // This is a placeholder for the actual implementation details
    // For example, you would implement methods like:
    // - buildScaleSpace()
    // - buildDoGSpace()
    // - findKeypointCandidates()
    // - refineKeypoints()
    // - assignOrientations()
    // - computeDescriptors()
    // Each method would contain the logic to perform the respective step in the SIFT algorithm
    std::vector<Keypoint> SIFT::detectAndCompute(const core::Tensor<core::float32> &src) {
        // Example input: 512 x 512 x 3 (RGB image)
        // Step 1: Build scale space
        // -> 4O, 5S (Octaves and Scales)
        // Octave 0: [blur00, blur01, blur02, blur03, blur04] -> Size: 512 x 512
        // Octave 1: [blur10, blur11, blur12, blur13, blur14] -> Size: 256 x 256
        // Octave 2: [blur20, blur21, blur22, blur23, blur24] -> Size: 128 x 128
        // Octave 3: [blur30, blur31, blur32, blur33, blur34] -> Size: 64 x 64
        //
        // Step 2: Build difference of Gaussians (DoG)
        // -> 4O, 4S (Octaves and Scales)
        // Octave 0: [dog00, dog01, dog02, dog03] -> Size: 512 x 512
        // Octave 1: [dog10, dog11, dog12, dog13] -> Size: 256 x 256
        // Octave 2: [dog20, dog21, dog22, dog23] -> Size: 128 x 128
        // Octave 3: [dog30, dog31, dog32, dog33] -> Size: 64 x 64

        // Step 1: Build scale space
        Layers scaleSpace = buildScaleSpace(src);

        // Step 2: Build difference of Gaussians
        Layers dogSpace = buildDoGSpace(scaleSpace);

        // Step 3: Find keypoint candidates
        std::vector<Keypoint> candidates = findKeypointCandidates(dogSpace);
        std::cout << "Found: " << candidates.size() << " keypoint candidates in the DoG space."
                  << std::endl;

        // Step 4: Refine keypoints and remove edge responses
        // auto keypoints = refineKeypoints(candidates, dogSpace);

        // Step 5: Assign orientations
        // assignOrientations(keypoints, scaleSpace);

        // Step 6: Compute descriptors
        // computeDescriptors(keypoints, scaleSpace);

        std::cout << "SIFT detector finished ✅" << std::endl;
        return {};
    }

}   // namespace processing
