//
// File        : core.hpp
// Author      : Hinsun
// Date        : 2025-06-19
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#ifndef CORE_HPP
#define CORE_HPP

#pragma once
#include <cstddef>
#include <cstdint>

#include "core/tensor.hpp"

namespace core {

// Define common types for tensors and basic data types
using int8 = std::int8_t;
using int16 = std::int16_t;
using int32 = std::int32_t;
using int64 = std::int64_t;

using uint8 = std::uint8_t;
using uint16 = std::uint16_t;
using uint32 = std::uint32_t;
using uint64 = std::uint64_t;

using float32 = float;
using float64 = double;

// Define tensor types for various data types
using TensorI8 = Tensor<core::int8>;
using TensorI16 = Tensor<core::int16>;
using TensorI32 = Tensor<core::int32>;
using TensorI64 = Tensor<core::int64>;

using TensorU8 = Tensor<core::uint8>;
using TensorU16 = Tensor<core::uint16>;
using TensorU32 = Tensor<core::uint32>;
using TensorU64 = Tensor<core::uint64>;

using TensorF32 = Tensor<core::float32>;
using TensorF64 = Tensor<core::float64>;

class Rect {
public:
    size_t x, y, width, height;
    explicit Rect(const size_t x = 0, const size_t y = 0, const size_t width = 0,
                  const size_t height = 0)
        : x(x), y(y), width(width), height(height) {}
};

}  // namespace core

#endif  // CORE_HPP
