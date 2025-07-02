//
// File        : orb.cpp
// Author      : Hinsun
// Date        : 2025-06-TODAY
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#include "core/common.hpp"
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

    std::vector<Keypoint> ORB::detectFASTKeypoints(const Layers &pyramid) {
        std::vector<Keypoint> keypoints;

        // Check Harris corner detector :D

        return keypoints;
    }

    void ORB::computeOrientations(std::vector<Keypoint> &keypoints, const Layers &pyramid) {}

    void ORB::computeDescriptors(std::vector<Keypoint> &keypoints, const Layers &pyramid) {}

    std::vector<Keypoint> ORB::detectAndCompute(const core::Tensor<core::float32> &src) {
        // Step 1: Build image pyramid
        const Layers pyramid = buildImagePyramid(src);

        // Step 2: Detect FAST keypoints
        std::vector<Keypoint> keypoints = detectFASTKeypoints(pyramid);

        // Step 3: Compute orientations for keypoints
        computeOrientations(keypoints, pyramid);

        // Step 4: Compute BRIEF descriptors for keypoints
        computeDescriptors(keypoints, pyramid);

        return keypoints;
    }

}   // namespace processing
