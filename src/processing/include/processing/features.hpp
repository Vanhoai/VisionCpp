//
// File        : features.hpp
// Author      : Hinsun
// Date        : 2025-06-26
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#ifndef FEATURES_HPP
#define FEATURES_HPP

/**
 * @brief Header for feature extraction operations.
 *
 * This file declares the some feature extractor class, which provides classes for extracting
 * various types of features from images.
 * - SIFT (Scale-Invariant Feature Transform)
 * - SURF (Speeded-Up Robust Features)
 * - ORB (Oriented FAST and Rotated BRIEF)
 */

#include <core/core.hpp>
#include <core/tensor.hpp>
#include <vector>

using Layers = std::vector<std::vector<core::TensorF32>>;

namespace processing {

    class Keypoint {
        public:
            float x, y;                      // Position in image
            float scale;                     // Scale (sigma)
            float angle;                     // Dominant orientation
            int octave;                      // Octave index
            int layer;                       // Layer within octave
            std::vector<float> descriptor;   // 128-dimensional descriptor

            Keypoint() : x(0), y(0), scale(0), angle(0), octave(0), layer(0) {
                descriptor.resize(128, 0.0f);
            }
    };

    /**
     * @brief Class for SIFT feature extraction.
     *
     * This class implements the SIFT algorithm for detecting and describing local features in
     * images.
     *
     * SIFT is a feature detection algorithm that is invariant to scale and rotation, making it
     * suitable for matching keypoints across different images. It identifies keypoints in an image
     * and computes descriptors that capture the local image structure around each keypoint.
     *
     * How SIFT works:
     * 1. Scale-space extrema detection: SIFT identifies potential keypoints by searching for
     *    extrema in a scale-space representation of the image.
     * 2. Keypoint localization: The algorithm refines the keypoints by fitting a model to the
     *    local image structure.
     * 3. Orientation assignment: Each keypoint is assigned a consistent orientation based on the
     *    local image gradient, which makes the descriptors rotation-invariant.
     * 4. Descriptor generation: SIFT computes a descriptor for each keypoint based on the local
     *    image gradients around the keypoint, resulting in a 128-dimensional vector.
     * 5. Keypoint matching: The descriptors can be used to match keypoints between different
     *    images, allowing for image registration, object recognition, and other applications.
     */
    class SIFT {
        private:
            // SIFT parameters
            static constexpr int NUM_OCTAVES = 4;
            static constexpr int NUM_SCALES = 5;   // s+3 where s=2
            static constexpr float SIGMA_0 = 1.6f;
            static constexpr float K = 1.414213562f;   // 2^(1/s)
            static constexpr float CONTRAST_THRESHOLD = 0.04f;
            static constexpr float EDGE_THRESHOLD = 10.0f;
            static constexpr float PEAK_RATIO = 0.8f;

            // Descriptor parameters
            static constexpr int DESC_WINDOW_SIZE = 16;
            static constexpr int DESC_HIST_BINS = 8;
            static constexpr int DESC_SIZE = 128;   // 4x4x8

            static core::TensorF32 gaussianBlur(const core::TensorF32 &src, float sigma);
            static core::TensorF32 downsample(const core::TensorF32 &src);
            static core::TensorF32 convolveHorizontal(const core::TensorF32 &src,
                                                      const std::vector<float> &kernel);
            static core::TensorF32 convolveVertical(const core::TensorF32 &src,
                                                    const std::vector<float> &kernel);
            static core::TensorF32 substract(const core::TensorF32 &a, const core::TensorF32 &b);
            static bool isLocalExtremum(int x, int y, const core::TensorF32 &current,
                                        const core::TensorF32 &below, const core::TensorF32 &above);

            static bool localizeKeypoint(Keypoint &kp, const Layers &dogSpace);
            static bool isEdgeResponse(const Keypoint &kp, const Layers &dogSpace);
            static std::vector<float> computeOrientations(const Keypoint &kp,
                                                          const Layers &scaleSpace);

            static void computeDescriptor(Keypoint &kp, const Layers &scaleSpace);

            static Layers buildScaleSpace(const core::TensorF32 &src);
            static Layers buildDoGSpace(const Layers &scaleSpace);
            static std::vector<Keypoint> findKeypointCandidates(const Layers &dogSpace);
            static std::vector<Keypoint> refineKeypoints(const std::vector<Keypoint> &candidates,
                                                         const Layers &dogSpace);

            static void assignOrientations(std::vector<Keypoint> &keypoints,
                                           const Layers &scaleSpace);
            static void computeDescriptors(std::vector<Keypoint> &keypoints,
                                           const Layers &scaleSpace);

        public:
            SIFT() = default;
            static std::vector<Keypoint> detectAndCompute(const core::TensorF32 &src);
    };

    /**
     * @brief Class for SURF feature extraction.
     *
     * This class implements the SURF algorithm for detecting and describing local features in
     * images.
     *
     * SURF is a fast and robust feature detection algorithm that is invariant to scale and
     * rotation, making it suitable for matching keypoints across different images. It is
     * designed to be faster than SIFT while maintaining similar performance in terms of
     * keypoint detection and matching.
     *
     * How SURF works:
     * 1. Interest point detection: Uses a Hessian matrix-based detector to find blob-like
     *    structures at various scales.
     * 2. Interest point localization: Refines keypoint locations using interpolation in
     *    scale-space.
     * 3. Orientation assignment: Assigns a reproducible orientation to each keypoint based
     *    on information from a circular region around the keypoint.
     * 4. Descriptor extraction: Constructs a descriptor based on Haar wavelet responses
     *    within a neighborhood of the keypoint, typically resulting in a 64 or 128-dimensional
     *    vector.
     * 5. Keypoint matching: Descriptors can be matched between images for applications
     *    like object recognition, image stitching, and 3D reconstruction.
     *
     * ⚠️ PATENT NOTICE:
     * The SURF algorithm is protected by patents in some jurisdictions. This implementation
     * is provided for educational and research purposes. For commercial use, please verify
     * patent status in your jurisdiction and obtain appropriate licenses if required.
     * See README.md for detailed licensing information and alternatives.
     *
     * @note For commercial applications, consider patent-free alternatives like ORB or AKAZE.
     */
    class SUFT {};

    /**
     * @brief Class for ORB feature extraction.
     *
     * This class implements the ORB (Oriented FAST and Rotated BRIEF) algorithm for detecting
     * and describing local features in images.
     *
     * ORB is a fast, patent-free alternative to SIFT and SURF, that combines the FAST keypoint
     * detector with the BRIEF descriptor. It provides good performance for real-time applications
     * and is particularly suitable for mobile and embedded systems.
     *
     * How ORB works:
     * 1. Keypoint Detection: Uses the FAST (Features from Accelerated Segment Test) algorithm
     *    to detect corner keypoints at multiple scale levels in an image pyramid.
     * 2. Harris Corner Measure: Applies Harris corner measure to rank FAST keypoints and
     *    select the most stable ones.
     * 3. Orientation Assignment: Computes the intensity centroid to assign a consistent
     *    orientation to each keypoint, making the descriptor rotation-invariant.
     * 4. Descriptor Computation: Uses a rotated version of the BRIEF (Binary Robust Independent
     *    Elementary Features) descriptor, creating a 256-bit binary descriptor.
     * 5. Keypoint Matching: Binary descriptors enable fast matching using Hamming distance.
     *
     * Key advantages of ORB:
     * - Patent-free and open source
     * - Fast computation suitable for real-time applications
     * - Binary descriptors enable efficient storage and matching
     * - Scale and rotation invariant
     * - Good performance on textured images
     *
     * @note ORB is particularly well-suited for SLAM, object tracking, and image matching
     * applications where speed is crucial.
     */
    class ORB {
        private:
            // ORB parameters
            static constexpr int NUM_LEVELS = 8;          // Number of pyramid levels
            static constexpr float SCALE_FACTOR = 1.2f;   // Scale factor between levels
            static constexpr int MAX_KEYPOINTS = 500;     // Maximum number of keypoints to detect
            static constexpr int FAST_THRESHOLD = 20;     // FAST corner threshold
            static constexpr int PATCH_SIZE = 31;         // Size of the BRIEF patch
            static constexpr int BORDER_SIZE = 16;        // Border size to avoid edge effects

        public:
            ORB() = default;

            static Layers buildImagePyramid(const core::TensorF32 &src);
            static std::vector<Keypoint> detectFASTKeypoints(const Layers &pyramid);
            static void computeOrientations(std::vector<Keypoint> &keypoints,
                                            const Layers &pyramid);
            static void computeDescriptors(std::vector<Keypoint> &keypoints, const Layers &pyramid);

            static std::vector<Keypoint> detectAndCompute(const core::TensorF32 &src);
    };

}   // namespace processing

#endif   // FEATURES_HPP
