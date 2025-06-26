//
// File        : filters.cpp
// Author      : Hinsun
// Date        : 2025-06-24
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#include "processing/filters.hpp"

#include <algorithm>

#include "processing/transformations.hpp"

namespace processing {

    static std::vector<std::vector<float>> createGaussianKernel(const int ksize,
                                                                const double sigma) {
        std::vector kernel(ksize, std::vector<float>(ksize));
        double sum = 0.0;
        const int half = ksize / 2;
        const double s2 = 2.0 * sigma * sigma;

        for (int y = -half; y <= half; ++y) {
            for (int x = -half; x <= half; ++x) {
                const float value = std::exp(-(x * x + y * y) / s2);
                kernel[y + half][x + half] = value;
                sum += value;
            }
        }

        // Normalize
        for (auto& row : kernel)
            for (auto& val : row) val /= sum;

        return kernel;
    }

    void Filters::gaussianBlur(const core::Tensor<core::float32>& src,
                               core::Tensor<core::float32>& dst, const int ksize) {
        if (src.empty())
            throw std::invalid_argument("Image tensor should not be empty");

        if (src.shape().size() != 3)
            throw std::invalid_argument(
                "Image tensor must have 3 dimensions (height, width, channels)");

        const int height = src.shape()[0];
        const int width = src.shape()[1];
        const int channels = src.shape()[2];

        const int half = ksize / 2;
        const double sigma = ksize / 3.0;

        const auto kernel = createGaussianKernel(ksize, sigma);
        dst = core::Tensor<core::float32>(height, width, channels);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < channels; c++) {
                    float sum = 0.0f;

                    for (int ky = -half; ky <= half; ++ky) {
                        for (int kx = -half; kx <= half; ++kx) {
                            const int iy = std::clamp(y + ky, 0, height - 1);
                            const int ix = std::clamp(x + kx, 0, width - 1);

                            sum += src.at(iy, ix, c) * kernel[ky + half][kx + half];
                        }
                    }

                    dst.at(y, x, c) = sum;
                }
            }
        }
    }

    void Filters::medianBlur(const core::Tensor<core::float32>& src,
                             core::Tensor<core::float32>& dst, const int ksize) {
        const int height = src.shape()[0];
        const int width = src.shape()[1];
        const int channels = src.shape()[2];
        const int half = ksize / 2;

        dst = core::Tensor<core::float32>(height, width, channels);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < channels; c++) {
                    std::vector<float> window;

                    for (int ky = -half; ky <= half; ++ky) {
                        for (int kx = -half; kx <= half; ++kx) {
                            const int iy = std::clamp(y + ky, 0, height - 1);
                            const int ix = std::clamp(x + kx, 0, width - 1);

                            window.push_back(src.at(iy, ix, c));
                        }
                    }

                    std::nth_element(window.begin(), window.begin() + window.size() / 2,
                                     window.end());
                    dst.at(y, x, c) = window[window.size() / 2];
                }
            }
        }
    }

    void Filters::unsharpMask(const core::Tensor<core::float32>& src,
                              core::Tensor<core::float32>& dst, const double sigma,
                              const double alpha) {
        core::Tensor<core::float32> blurred;
        gaussianBlur(src, blurred, static_cast<int>(std::round(sigma * 3) * 2 + 1));
        const int height = src.shape()[0];
        const int width = src.shape()[1];
        const int channels = src.shape()[2];

        dst = core::Tensor<core::float32>(height, width, channels);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int c = 0; c < channels; ++c) {
                    const float original = src.at(y, x, c);
                    const float blurredVal = blurred.at(y, x, c);
                    float sharp = original + alpha * (original - blurredVal);

                    dst.at(y, x, c) = std::clamp(sharp, 0.0f, 255.0f);   // Clamp to valid range
                }
            }
        }
    }

    void Filters::bilateralFilter(const core::Tensor<core::float32>& src,
                                  core::Tensor<core::float32>& dst, const int ksize,
                                  const double sigmaSpatial, const double sigmaRange) {
        if (src.empty())
            throw std::invalid_argument("Image tensor should not be empty");

        if (src.shape().size() != 3)
            throw std::invalid_argument(
                "Image tensor must have 3 dimensions (height, width, channels)");

        const int height = src.shape()[0];
        const int width = src.shape()[1];
        const int channels = src.shape()[2];

        const int half = ksize / 2;
        const double spatialCoeff = -0.5 / (sigmaSpatial * sigmaSpatial);
        const double rangeCoeff = -0.5 / (sigmaRange * sigmaRange);

        dst = core::Tensor<core::float32>(height, width, channels);

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int c = 0; c < channels; ++c) {
                    const float center = src.at(y, x, c);
                    float sum = 0.0f;
                    float weightSum = 0.0f;

                    for (int ky = -half; ky <= half; ++ky) {
                        for (int kx = -half; kx <= half; ++kx) {
                            const int iy = std::clamp(y + ky, 0, height - 1);
                            const int ix = std::clamp(x + kx, 0, width - 1);

                            const float neighbor = src.at(iy, ix, c);
                            const float spatialDist2 = ky * ky + kx * kx;
                            const float rangeDist2 = (center - neighbor) * (center - neighbor);

                            const float weight =
                                std::exp(spatialCoeff * spatialDist2 + rangeCoeff * rangeDist2);

                            sum += weight * neighbor;
                            weightSum += weight;
                        }
                    }

                    dst.at(y, x, c) = sum / weightSum;
                }
            }
        }
    }

    void Filters::sobel(const core::Tensor<core::float32>& src, core::Tensor<core::float32>& gradX,
                        core::Tensor<core::float32>& gradY, const SobelDirection direction) {
        if (src.empty())
            throw std::invalid_argument("Image tensor should not be empty");

        if (src.dimensions() != 2)
            throw std::invalid_argument("This algorithms only supports single channel images");

        const int height = src.shape()[0];
        const int width = src.shape()[1];

        gradX = core::Tensor<core::float32>(height, width);
        gradY = core::Tensor<core::float32>(height, width);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                gradX.at(y, x) = 0.0f;
                gradY.at(y, x) = 0.0f;
            }
        }

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                float gx = 0.0f, gy = 0.0f;

                for (int j = -1; j <= 1; j++) {
                    for (int i = -1; i <= 1; i++) {
                        const float pixel = src.at(y + j, x + i);
                        gx += kernelX[j + 1][i + 1] * pixel;
                        gy += kernelY[j + 1][i + 1] * pixel;
                    }
                }

                // Apply direction filter
                switch (direction) {
                    case SobelDirection::X:
                        gradX.at(y, x) = gx;
                        gradY.at(y, x) = 0.0f;
                        break;
                    case SobelDirection::Y:
                        gradX.at(y, x) = 0.0f;
                        gradY.at(y, x) = gy;
                        break;
                    case SobelDirection::XY:
                    default:
                        gradX.at(y, x) = gx;
                        gradY.at(y, x) = gy;
                        break;
                }
            }
        }
    }

    void Filters::computeMagnitudeDirection(const core::Tensor<core::float32>& gradX,
                                            const core::Tensor<core::float32>& gradY,
                                            core::Tensor<core::float32>& magnitude,
                                            core::Tensor<core::float32>& direction) {
        const int height = gradX.shape()[0];
        const int width = gradX.shape()[1];

        magnitude = core::Tensor<core::float32>(height, width);
        direction = core::Tensor<core::float32>(height, width);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                const float gx = gradX.at(y, x);
                const float gy = gradY.at(y, x);

                // Compute magnitude
                const float mag = std::sqrt(gx * gx + gy * gy);
                magnitude.at(y, x) = mag;

                // Compute direction (in degrees)
                float angle = std::atan2(gy, gx) * 180.0f / M_PI;
                if (angle < 0.0f)
                    angle += 360.0f;

                direction.at(y, x) = angle;
            }
        }
    }

    void calculateGradient(const core::Tensor<core::float32>& src,
                           core::Tensor<core::float32>& magnitude,
                           core::Tensor<core::float32>& direction) {
        if (src.empty())
            throw std::invalid_argument("Image tensor should not be empty");

        if (src.dimensions() != 2)
            throw std::invalid_argument(
                "Please convert image to grayscale before using this function");

        const int height = src.shape()[0];
        const int width = src.shape()[1];

        magnitude = core::Tensor<core::float32>(height, width);
        direction = core::Tensor<core::float32>(height, width);

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                float gx = 0.0f, gy = 0.0f;

                // Apply Sobel operators
                for (int j = -1; j <= 1; j++) {
                    for (int i = -1; i <= 1; i++) {
                        const core::float32 pixel = src.at(y + j, x + i);
                        gx += pixel * kernelX[j + 1][i + 1];
                        gy += pixel * kernelY[j + 1][i + 1];
                    }
                }

                float mag = std::sqrt(gx * gx + gy * gy);
                float angle = std::atan2(gy, gx) * 180.0f / M_PI;
                if (angle < 0.0f)
                    angle += 180.0f;

                magnitude.at(y, x) = std::min(255.0f, mag);
                direction.at(y, x) = angle;
            }
        }
    }

    void nonMaximumSuppression(const core::Tensor<core::float32>& magnitude,
                               const core::Tensor<core::float32>& direction,
                               core::Tensor<core::float32>& suppressed) {
        const int height = magnitude.shape()[0];
        const int width = magnitude.shape()[1];

        suppressed = core::Tensor<core::float32>(height, width);
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                const core::float32 angle = direction.at(y, x);
                const core::float32 mag = magnitude.at(y, x);

                core::float32 neighbor1 = 0.0f, neighbor2 = 0.0f;

                // Determine neighbors based on gradient direction
                if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180)) {
                    // Horizontal direction (0 degrees)
                    neighbor1 = magnitude.at(y, x + 1);
                    neighbor2 = magnitude.at(y, x - 1);
                } else if (angle >= 22.5 && angle < 67.5) {
                    // Diagonal direction (45 degrees)
                    neighbor1 = magnitude.at(y - 1, x + 1);
                    neighbor2 = magnitude.at(y + 1, x - 1);
                } else if (angle >= 67.5 && angle < 112.5) {
                    // Vertical direction (90 degrees)
                    neighbor1 = magnitude.at(y - 1, x);
                    neighbor2 = magnitude.at(y + 1, x);
                } else if (angle >= 112.5 && angle < 157.5) {
                    // Diagonal direction (135 degrees)
                    neighbor1 = magnitude.at(y - 1, x - 1);
                    neighbor2 = magnitude.at(y + 1, x + 1);
                }

                // Keep pixel if it's a local maximum
                if (mag >= neighbor1 && mag >= neighbor2)
                    suppressed.at(y, x) = mag;
                else
                    suppressed.at(y, x) = 0.0f;   // Suppress non-maximum pixels
            }
        }
    }

    void connectWeakEdges(core::Tensor<core::float32>& dst, core::Tensor<core::int8>& visited,
                          const int u, const int v) {
        const int height = dst.shape()[0];
        const int width = dst.shape()[1];
        if (u < 0 || u >= height || v < 0 || v >= width || visited.at(u, v) == 1)
            return;

        visited.at(u, v) = 1;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0)
                    continue;

                int dx = u + i, dy = v + j;
                if (dx >= 0 && dx < height && dy >= 0 && dy < width && visited.at(dx, dy) == 0 &&
                    dst.at(dx, dy) == 128.0f) {
                    dst.at(dx, dy) = 255.0f;
                    connectWeakEdges(dst, visited, dx, dy);
                }
            }
        }
    }

    void hysteresisThresholding(const core::Tensor<core::float32>& suppressed,
                                core::Tensor<core::float32>& dst, const double lowThreshold,
                                const double highThreshold) {
        const size_t height = suppressed.shape()[0];
        const size_t width = suppressed.shape()[1];

        dst = core::Tensor<core::float32>(height, width);
        core::Tensor<core::int8> visited({height, width}, 0);

        // Step 1: Mark strong and weak edges
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                if (const float pixel = suppressed.at(y, x); pixel >= highThreshold)
                    dst.at(y, x) = 255.0f;
                else if (pixel >= lowThreshold)
                    dst.at(y, x) = 128.0f;
                else
                    dst.at(y, x) = 0.0f;
            }
        }

        // Step 2: Recursively connect weak edges to strong edges
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                if (dst.at(y, x) == 255.0f && visited.at(y, x) == 0) {
                    connectWeakEdges(dst, visited, y, x);
                }
            }
        }

        // Step 3: Remove unconnected weak edges
        for (size_t y = 0; y < height; y++) {
            for (size_t x = 0; x < width; x++) {
                if (dst.at(y, x) == 128.0f)
                    dst.at(y, x) = 0.0f;
            }
        }
    }

    void Filters::canny(const core::Tensor<core::float32>& src, core::Tensor<core::float32>& dst,
                        const double lowThreshold, const double highThreshold, const int ksize) {
        // 1. Noise Reduction
        core::Tensor<core::float32> blurred;
        gaussianBlur(src, blurred, ksize);
        core::Tensor<core::float32> grayscale;
        Transformations::convertToGrayScale(blurred, grayscale);

        // 2. Gradient Calculation
        core::Tensor<core::float32> magnitude, direction;
        calculateGradient(grayscale, magnitude, direction);

        // 3. Non-Maximum Suppression
        core::Tensor<core::float32> suppressed;
        nonMaximumSuppression(magnitude, direction, suppressed);

        // 4: Hysteresis Thresholding
        hysteresisThresholding(suppressed, dst, lowThreshold, highThreshold);
    }

}   // namespace processing