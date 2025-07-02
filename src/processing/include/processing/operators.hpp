//
// File        : operators.hpp
// Author      : Hinsun
// Date        : 2025-06-TODAY
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include <vector>

#include "core/core.hpp"

namespace processing {

    /**
     * @brief Performs some operation for tensors class
     *
     * This file contains operator for tensor class such as:
     * - convolveHorizontal
     * - convolveVertical
     * - gaussianBlur (uses convolveHorizontal and convolveVertical)
     * - downsample (decreases the size of the tensor by half)
     * - substract (subtracts two tensors)
     */

    core::TensorF32 convolveHorizontal(const core::TensorF32& src,
                                       const std::vector<core::float32>& kernel);

    core::TensorF32 convolveVertical(const core::TensorF32& src,
                                     const std::vector<core::float32>& kernel);

    core::TensorF32 gaussianBlur(const core::TensorF32& src, core::float32 sigma);

    core::TensorF32 downsample(const core::TensorF32& src);

    core::TensorF32 substract(const core::TensorF32& a, const core::TensorF32& b);

}   // namespace processing

#endif   // OPERATORS_HPP
