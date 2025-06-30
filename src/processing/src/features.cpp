//
// File        : features.cpp
// Author      : Hinsun
// Date        : 2025-06-26
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#include "processing/features.hpp"

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

    bool SIFT::localizeKeypoint(Keypoint &kp, const Layers &dogSpace) {
        const auto &dog = dogSpace[kp.octave][kp.layer];
        int x = static_cast<int>(kp.x / std::pow(2.0f, kp.octave));
        int y = static_cast<int>(kp.y / std::pow(2.0f, kp.octave));

        const int height = dog.shape()[0];
        const int width = dog.shape()[1];

        if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
            return false;

        // Compute derivatives
        float dx = (dog.at(y, x + 1) - dog.at(y, x - 1)) * 0.5f;
        float dy = (dog.at(y + 1, x) - dog.at(y - 1, x)) * 0.5f;

        // Compute second derivatives (Hessian)
        float dxx = dog.at(y, x + 1) + dog.at(y, x - 1) - 2 * dog.at(y, x);
        float dyy = dog.at(y + 1, x) + dog.at(y - 1, x) - 2 * dog.at(y, x);

        float dxy = (dog.at(y + 1, x + 1) - dog.at(y + 1, x - 1) - dog.at(y - 1, x + 1) +
                     dog.at(y - 1, x - 1)) *
                    0.25f;

        // Solve linear system to find offset
        float det = dxx * dyy - dxy * dxy;
        if (std::abs(det) < 1e-6)
            return false;

        float offset_x = -(dyy * dx - dxy * dy) / det;
        float offset_y = -(dxx * dy - dxy * dx) / det;

        // Check if offset is reasonable
        if (std::abs(offset_x) > 0.5f || std::abs(offset_y) > 0.5f)
            return false;

        // Update keypoint position
        kp.x += offset_x * std::pow(2.0f, kp.octave);
        kp.y += offset_y * std::pow(2.0f, kp.octave);

        // Check contrast
        float contrast = dog.at(y, x) + 0.5f * (dx * offset_x + dy * offset_y);
        if (std::abs(contrast) < CONTRAST_THRESHOLD)
            return false;

        return true;
    }

    bool SIFT::isEdgeResponse(const Keypoint &kp, const Layers &dogSpace) {
        const auto &dog = dogSpace[kp.octave][kp.layer];
        int x = static_cast<int>(kp.x / std::pow(2.0f, kp.octave));
        int y = static_cast<int>(kp.y / std::pow(2.0f, kp.octave));

        const int height = dog.shape()[0];
        const int width = dog.shape()[1];

        if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
            return true;

        // Compute Hessian matrix
        float dxx = dog.at(y, x + 1) + dog.at(y, x - 1) - 2 * dog.at(y, x);
        float dyy = dog.at(y + 1, x) + dog.at(y - 1, x) - 2 * dog.at(y, x);
        float dxy = (dog.at(y + 1, x + 1) - dog.at(y + 1, x - 1) - dog.at(y - 1, x + 1) +
                     dog.at(y - 1, x - 1)) *
                    0.25f;

        float trace = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;

        if (det <= 0)
            return true;

        float ratio = trace * trace / det;
        float threshold = (EDGE_THRESHOLD + 1) * (EDGE_THRESHOLD + 1) / EDGE_THRESHOLD;

        return ratio > threshold;
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
            for (int scale = 1; scale < NUM_SCALES - 2; scale++) {
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

    std::vector<Keypoint> SIFT::refineKeypoints(const std::vector<Keypoint> &candidates,
                                                const Layers &dogSpace) {
        std::vector<Keypoint> refined;
        for (const auto &candidate : candidates) {
            Keypoint kp = candidate;
            // Subpixel localization using Taylor expansion
            if (localizeKeypoint(kp, dogSpace)) {
                // Remove edge responses using Hessian
                if (!isEdgeResponse(kp, dogSpace))
                    refined.push_back(kp);
            }
        }

        return refined;
    }

    std::vector<float> SIFT::computeOrientations(const Keypoint &kp, const Layers &scaleSpace) {
        const auto &image = scaleSpace[kp.octave][kp.layer];
        int x = static_cast<int>(kp.x / std::pow(2.0f, kp.octave));
        int y = static_cast<int>(kp.y / std::pow(2.0f, kp.octave));
        float sigma = 1.5f * kp.scale / std::pow(2.0f, kp.octave);
        int radius = static_cast<int>(3 * sigma);

        const int height = image.shape()[0];
        const int width = image.shape()[1];

        // Orientation histogram (36 bins)
        std::vector<float> hist(36, 0.0f);
        for (int dy = -radius; dy <= radius; dy++) {
            for (int dx = -radius; dx <= radius; dx++) {
                int px = x + dx, py = y + dy;
                if (px < 1 || px >= width - 1 || py < 1 || py >= height - 1)
                    continue;

                // Compute gradient
                float gx = image.at(py, px + 1) - image.at(py, px - 1);
                float gy = image.at(py + 1, px) - image.at(py - 1, px);
                float magnitude = std::sqrt(gx * gx + gy * gy);
                float angle = std::atan2(gy, gx);
                // Gaussian weighting
                float weight = std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
                // Add to histogram
                int bin = static_cast<int>((angle + M_PI) * 36 / (2 * M_PI)) % 36;
                hist[bin] += magnitude * weight;
            }
        }

        // Smooth histogram
        for (int i = 0; i < 6; i++) {
            std::vector<float> smoothed(36);
            for (int j = 0; j < 36; j++)
                smoothed[j] = (hist[(j - 1 + 36) % 36] + hist[j] + hist[(j + 1) % 36]) / 3.0f;

            hist = smoothed;
        }

        // Find peaks
        std::vector<float> orientations;
        float maxVal = *std::max_element(hist.begin(), hist.end());

        for (int i = 0; i < 36; i++) {
            if (hist[i] > 0.8f * maxVal) {
                // Parabolic interpolation for sub-bin accuracy
                float prev = hist[(i - 1 + 36) % 36];
                float curr = hist[i];
                float next = hist[(i + 1) % 36];

                float offset = 0.5f * (prev - next) / (prev - 2 * curr + next);
                float angle = (i + offset) * 2 * M_PI / 36 - M_PI;
                orientations.push_back(angle);
            }
        }

        return orientations;
    }

    void SIFT::assignOrientations(std::vector<Keypoint> &keypoints, const Layers &scaleSpace) {
        std::vector<Keypoint> orientedKeypoints;

        for (auto &kp : keypoints) {
            auto orientations = computeOrientations(kp, scaleSpace);

            // Create keypoint for dominant orientation
            if (!orientations.empty()) {
                kp.angle = orientations[0];
                orientedKeypoints.push_back(kp);

                // Create additional keypoints for secondary orientations
                for (size_t i = 1; i < orientations.size(); i++) {
                    if (orientations[i] > PEAK_RATIO * orientations[0]) {
                        Keypoint newKp = kp;
                        newKp.angle = orientations[i];
                        orientedKeypoints.push_back(newKp);
                    }
                }
            }
        }
    }

    void SIFT::computeDescriptors(std::vector<Keypoint> &keypoints, const Layers &scaleSpace) {
        for (auto &kp : keypoints) computeDescriptor(kp, scaleSpace);
    }

    void SIFT::computeDescriptor(Keypoint &kp, const Layers &scaleSpace) {
        const auto &image = scaleSpace[kp.octave][kp.layer];

        int x = static_cast<int>(kp.x / std::pow(2.0f, kp.octave));
        int y = static_cast<int>(kp.y / std::pow(2.0f, kp.octave));

        float scale = kp.scale / std::pow(2.0f, kp.octave);
        float cos_angle = std::cos(kp.angle);
        float sin_angle = std::sin(kp.angle);
        // 4x4 subregions, each 4x4 pixels
        int hist_size = 8;   // 8 orientation bins

        std::vector descriptor(4 * 4 * hist_size, 0.0f);

        const int height = image.shape()[0];
        const int width = image.shape()[1];

        // Sample points in 16x16 window around keypoint
        for (int i = -8; i < 8; i++) {
            for (int j = -8; j < 8; j++) {
                // Rotate sample point
                const float dx = i * cos_angle - j * sin_angle;
                const float dy = i * sin_angle + j * cos_angle;
                const int px = x + static_cast<int>(dx * scale);
                const int py = y + static_cast<int>(dy * scale);

                if (px < 1 || px >= height - 1 || py < 1 || py >= width - 1)
                    continue;

                // Compute gradient
                const float gx = image.at(py, px + 1) - image.at(py, px - 1);
                const float gy = image.at(py + 1, px) - image.at(py - 1, px);
                const float magnitude = std::sqrt(gx * gx + gy * gy);
                const float angle = std::atan2(gy, gx) - kp.angle;

                // Gaussian weighting
                const float weight = std::exp(-(i * i + j * j) / (2 * (8 * scale) * (8 * scale)));

                // Determine which 4x4 subregion this sample belongs to
                int sub_x = (i + 8) / 4;   // 0-3
                int sub_y = (j + 8) / 4;   // 0-3

                if (sub_x >= 4)
                    sub_x = 3;

                if (sub_y >= 4)
                    sub_y = 3;

                // Determine orientation bin
                const int bin =
                    static_cast<int>((angle + M_PI) * hist_size / (2 * M_PI)) % hist_size;

                // Add to descriptor
                int desc_idx = (sub_y * 4 + sub_x) * hist_size + bin;
                descriptor[desc_idx] += magnitude * weight;
            }
        }

        // Normalize descriptor
        float norm = 0.0f;
        for (float val : descriptor) norm += val * val;

        norm = std::sqrt(norm);
        if (norm > 0) {
            for (float &val : descriptor) {
                val /= norm;
                // Threshold to 0.2 and renormalize (illumination invariance)
                val = std::min(val, 0.2f);
            }

            // Renormalize
            norm = 0.0f;
            for (float val : descriptor) norm += val * val;

            norm = std::sqrt(norm);
            if (norm > 0)
                for (float &val : descriptor) val /= norm;
        }

        kp.descriptor = descriptor;
    }

    std::vector<Keypoint> SIFT::detectAndCompute(const core::Tensor<core::float32> &src) {
        // Step 1: Build scale space
        const Layers scaleSpace = buildScaleSpace(src);

        // Step 2: Build difference of Gaussians
        const Layers dogSpace = buildDoGSpace(scaleSpace);

        // Step 3: Find keypoint candidates
        const std::vector<Keypoint> candidates = findKeypointCandidates(dogSpace);

        // Step 4: Refine keypoints and remove edge responses
        std::vector<Keypoint> keypoints = refineKeypoints(candidates, dogSpace);

        // Step 5: Assign orientations
        assignOrientations(keypoints, scaleSpace);

        // Step 6: Compute descriptors
        // computeDescriptors(keypoints, scaleSpace);

        return keypoints;
    }

}   // namespace processing
