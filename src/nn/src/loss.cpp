//
// Created by VanHoai on 8/6/25.
//

#include "nn/loss.hpp"

namespace nn {
double CrossEntropyLoss::operator()(Eigen::MatrixXd &Y, Eigen::MatrixXd &A) {
    const Eigen::MatrixXd clipped = A.array().max(epsilon).min(1 - epsilon);
    Eigen::MatrixXd log = clipped.array().log();
    const Eigen::MatrixXd loss = Y.array() * log.array();
    const Eigen::VectorXd rowMax = loss.rowwise().sum();
    return -rowMax.mean();
}

// FIXME: A - Y only works if the activation function is softmax
Eigen::MatrixXd CrossEntropyLoss::derivative(Eigen::MatrixXd &Y, Eigen::MatrixXd &A) {
    return A.array() - Y.array();
}

}  // namespace nn