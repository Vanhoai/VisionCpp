//
// File        : benchmark.cpp
// Author      : Hinsun
// Date        : 2025-06-19
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#include "benchmark.hpp"

#include <opencv2/opencv.hpp>

#include "core/core.hpp"
#include "core/tensor.hpp"

int main() {
    const core::Tensor<core::float32> A({
        {3.4, 5.6, 7.8},
        {1.2, 3.4, 5.6},
        {9.0, 1.2, 3.4},
    });

    std::cout << A << std::endl;

    const core::Tensor B = core::Tensor<core::float32>::cast<core::int32>(A);
    std::cout << B << std::endl;

    return EXIT_SUCCESS;
}
