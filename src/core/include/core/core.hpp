//
// File        : core.hpp
// Author      : Hinsun
// Date        : 2025-06-19
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#ifndef MACROS_HPP
#define MACROS_HPP

#pragma once
#include <cstddef>
#include <cstdint>

namespace core {

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

    class Rect {
        public:
            size_t x, y, width, height;
            explicit Rect(const size_t x = 0, const size_t y = 0, const size_t width = 0,
                          const size_t height = 0)
                : x(x), y(y), width(width), height(height) {}
    };

}   // namespace core

#endif   // MACROS_HPP
