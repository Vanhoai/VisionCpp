//
// Created by VanHoai on 9/6/25.
//

#include "nn/layer.hpp"

#include <iomanip>
#include <iostream>

namespace nn {
Layer::Layer(const int inputDimension, const int outputDimension,
             std::unique_ptr<Activation> activation) {
    this->inputDimension = inputDimension;
    this->outputDimension = outputDimension;
    this->activation = move(activation);
    // initialize weights and biases

    W = Eigen::MatrixXd::Random(inputDimension, outputDimension);
    b = Eigen::MatrixXd::Zero(1, outputDimension);

    // initialize props for backpropagation
    // notice: when use X= R(X x d) => Z & A will be change
    Z = Eigen::MatrixXd::Zero(outputDimension, 1);  // Z = W^T * X + b
    A = Eigen::MatrixXd::Zero(outputDimension, 1);  // A = activation(Z)

    dW = Eigen::MatrixXd::Zero(inputDimension, outputDimension);
    db = Eigen::MatrixXd::Zero(1, outputDimension);
}

Eigen::MatrixXd Layer::forward(const Eigen::MatrixXd &X) {
    const long N = X.rows();
    this->Z = Eigen::MatrixXd::Zero(N, outputDimension);
    this->A = Eigen::MatrixXd::Zero(N, outputDimension);

    const Eigen::MatrixXd B = b.replicate(N, 1);
    this->Z = X * W + B;
    this->A = (*activation)(Z);
    return this->A;
}

}  // namespace nn