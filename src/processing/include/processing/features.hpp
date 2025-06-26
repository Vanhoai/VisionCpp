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

namespace processing {

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
     * extrema in a scale-space representation of the image.
     * 2. Keypoint localization: The algorithm refines the keypoints by fitting a model to the
     * local image structure.
     * 3. Orientation assignment: Each keypoint is assigned a consistent orientation based on the
     * local image gradient, which makes the descriptors rotation-invariant.
     * 4. Descriptor generation: SIFT computes a descriptor for each keypoint based on the local
     * image gradients around the keypoint, resulting in a 128-dimensional vector.
     * 5. Keypoint matching: The descriptors can be used to match keypoints between different
     * images, allowing for image registration, object recognition, and other applications.
     *
     */
    class SIFT {
        public:
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
     * structures at various scales.
     * 2. Interest point localization: Refines keypoint locations using interpolation in
     * scale-space.
     * 3. Orientation assignment: Assigns a reproducible orientation to each keypoint based
     * on information from a circular region around the keypoint.
     * 4. Descriptor extraction: Constructs a descriptor based on Haar wavelet responses
     * within a neighborhood of the keypoint, typically resulting in a 64 or 128-dimensional
     * vector.
     * 5. Keypoint matching: Descriptors can be matched between images for applications
     * like object recognition, image stitching, and 3D reconstruction.
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

}   // namespace processing

#endif   // FEATURES_HPP
