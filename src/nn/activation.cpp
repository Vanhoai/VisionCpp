//
// Created by VanHoai on 8/6/25.
//

#include "activation.hpp"
#include <iostream>

namespace nn {
    // f(x) = max(0, x)
    MatrixXd ReLUActivation::operator()(MatrixXd &X) {
        return X.array().max(0.0);
    }

    // f'(x) = 1 if x > 0, else 0
    MatrixXd ReLUActivation::derivative(MatrixXd &X) {
        return (X.array() > 0.0).cast<double>();
    }

    // f(x) = 1 / (1 + exp(-x))
    MatrixXd SigmoidActivation::operator()(MatrixXd &X) {
        return 1.0 / (1.0 + (-X.array()).exp());
    }

    // f'(x) = f(x) * (1 - f(x))
    MatrixXd SigmoidActivation::derivative(MatrixXd &X) {
        return X.array() * (1.0 - X.array());
    }

    // f(x) = exp(x) / sum(exp(x))
    MatrixXd SoftmaxActivation::operator()(MatrixXd &X) {
        // exp = np.exp(X - np.max(X, axis=1))
        // return exp / np.sum(exp, axis=1)

        // const VectorXd maxRow = X.rowwise().maxCoeff();
        // const MatrixXd maxRowExpanded = maxRow.replicate(1, X.cols());
        // const MatrixXd exp = (X.array() - maxRowExpanded.array()).exp();
        //
        // const VectorXd sumExp = exp.rowwise().sum();
        // const MatrixXd sumExpExpanded = sumExp.replicate(1, X.cols());
        // return exp.array() / sumExpExpanded.array();

        MatrixXd Y(X.rows(), X.cols());
        for (int i = 0; i < X.rows(); ++i) {
            double maxRow = X.row(i).maxCoeff();
            RowVectorXd shifted = X.row(i).array() - maxRow;
            RowVectorXd expRow = shifted.array().exp();
            double sumExp = expRow.sum();
            Y.row(i) = expRow / sumExp;
        }

        return Y;
    }

    // f(x) = 1 (because softmax and loss derivative will be combined and
    // calculate in loss derivation)
    MatrixXd SoftmaxActivation::derivative(MatrixXd &X) {
        return MatrixXd::Ones(X.rows(), X.cols());
    }

}   // namespace nn