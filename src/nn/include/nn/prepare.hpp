//
// Created by VanHoai on 10/6/25.
//

#ifndef PREPARE_HPP
#define PREPARE_HPP

#include <Eigen/Core>

namespace nn {
void prepare(int N, int groups, int d, Eigen::MatrixXd &X, Eigen::MatrixXd &Y);
}

#endif  // PREPARE_HPP
