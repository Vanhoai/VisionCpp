//
// File        : benchmark.cpp
// Author      : Hinsun
// Date        : 2025-06-19
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#include "benchmark.hpp"

#include <iostream>

int main() {
    if (__cplusplus == 202302L)
        std::cout << "C++23";
    else if (__cplusplus == 202002L)
        std::cout << "C++20";
    else if (__cplusplus == 201703L)
        std::cout << "C++17";
    else if (__cplusplus == 201402L)
        std::cout << "C++14";
    else if (__cplusplus == 201103L)
        std::cout << "C++11";
    else if (__cplusplus == 199711L)
        std::cout << "C++98";

    return EXIT_SUCCESS;
}
