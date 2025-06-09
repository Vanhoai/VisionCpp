//
// Created by VanHoai on 8/6/25.
//

#include "loss.hpp"

namespace nn {
    double CrossEntropyLoss::operator()(MatrixXd &Y, MatrixXd &A) {
        // double epsilon = 1e-15;
        // A = np.clip(A, epsilon, 1 - epsilon)
        // return -np.mean(np.sum(Y * np.log(A), axis=1))

        MatrixXd clipped = A.array().max(epsilon).min(1 - epsilon);
        A = clipped.array().log();
        const MatrixXd loss = Y.array() * A.array();
        return -loss.sum() / Y.rows();
    }

    MatrixXd CrossEntropyLoss::derivative(MatrixXd &Y, MatrixXd &A) {
        return A.array() - Y.array();
    }

}   // namespace nn