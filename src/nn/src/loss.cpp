//
// Created by VanHoai on 8/6/25.
//

#include "nn/loss.hpp"

namespace nn {
    double CrossEntropyLoss::operator()(MatrixXd &Y, MatrixXd &A) {
        const MatrixXd clipped = A.array().max(epsilon).min(1 - epsilon);
        MatrixXd log = clipped.array().log();
        const MatrixXd loss = Y.array() * log.array();
        const VectorXd rowMax = loss.rowwise().sum();
        return -rowMax.mean();
    }

    // FIXME: A - Y only works if the activation function is softmax
    MatrixXd CrossEntropyLoss::derivative(MatrixXd &Y, MatrixXd &A) {
        return A.array() - Y.array();
    }

}   // namespace nn