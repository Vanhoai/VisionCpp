//
// File        : benchmark.cpp
// Author      : Hinsun
// Date        : 2025-06-19
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#include "benchmark.hpp"

#include <opencv2/opencv.hpp>

#include "core/common.hpp"

int main() {
    const core::Tensor<double> A({
        {1, 4},
        {3, 3},
    });

    const core::Tensor<double> B({
        {2, 2},
        {2, 2},
    });

    const core::Tensor<int> C = core::Tensor<int>::zeros({4});
    return EXIT_SUCCESS;
}
