//
// Created by VanHoai on 8/6/25.
//

#include "nn/activation.hpp"

namespace nn {
    // f(x) = max(0, x)
    Eigen::MatrixXd ReLUActivation::operator()(Eigen::MatrixXd &X) { return X.array().max(0.0); }

    // f'(x) = 1 if x > 0, else 0
    Eigen::MatrixXd ReLUActivation::derivative(Eigen::MatrixXd &X) {
        return (X.array() > 0.0).cast<double>();
    }

    // f(x) = 1 / (1 + exp(-x))
    Eigen::MatrixXd SigmoidActivation::operator()(Eigen::MatrixXd &X) {
        return 1.0 / (1.0 + (-X.array()).exp());
    }

    // f'(x) = f(x) * (1 - f(x))
    Eigen::MatrixXd SigmoidActivation::derivative(Eigen::MatrixXd &X) {
        return X.array() * (1.0 - X.array());
    }

    // f(x) = exp(x) / sum(exp(x))
    Eigen::MatrixXd SoftmaxActivation::operator()(Eigen::MatrixXd &X) {
        // exp = np.exp(X - np.max(X, axis=1))
        // return exp / np.sum(exp, axis=1)

        // const VectorXd maxRow = X.rowwise().maxCoeff();
        // const Eigen::MatrixXd maxRowExpanded = maxRow.replicate(1, X.cols());
        // const Eigen::MatrixXd exp = (X.array() - maxRowExpanded.array()).exp();
        //
        // const VectorXd sumExp = exp.rowwise().sum();
        // const Eigen::MatrixXd sumExpExpanded = sumExp.replicate(1, X.cols());
        // return exp.array() / sumExpExpanded.array();

        Eigen::MatrixXd Y(X.rows(), X.cols());
        for (int i = 0; i < X.rows(); ++i) {
            double maxRow = X.row(i).maxCoeff();
            Eigen::RowVectorXd shifted = X.row(i).array() - maxRow;
            Eigen::RowVectorXd expRow = shifted.array().exp();
            double sumExp = expRow.sum();
            Y.row(i) = expRow / sumExp;
        }

        return Y;
    }

    // f(x) = 1 (because softmax and loss derivative will be combined and
    // calculate in loss derivation)
    Eigen::MatrixXd SoftmaxActivation::derivative(Eigen::MatrixXd &X) {
        return Eigen::MatrixXd::Ones(X.rows(), X.cols());
    }

}   // namespace nn
