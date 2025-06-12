//
// Created by VanHoai on 10/6/25.
//

#ifndef PREPARE_HPP
#define PREPARE_HPP

#include <Eigen/Core>
using namespace Eigen;

namespace nn {
    void prepare(const int N, const int groups, const int d, MatrixXd &X,
             MatrixXd &Y);
}

#endif   // PREPARE_HPP
