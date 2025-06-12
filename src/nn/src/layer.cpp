//
// Created by VanHoai on 9/6/25.
//

#include "nn/layer.hpp"
#include <iomanip>
#include <iostream>

namespace nn {
    Layer::Layer(const int inputDimension, const int outputDimension,
                 unique_ptr<Activation> activation) {
        this->inputDimension = inputDimension;
        this->outputDimension = outputDimension;
        this->activation = move(activation);
        // initialize weights and biases

        W = MatrixXd::Random(inputDimension, outputDimension);
        b = MatrixXd::Zero(1, outputDimension);

        // initialize props for backpropagation
        // notice: when use X= R(X x d) => Z & A will be change
        Z = MatrixXd::Zero(outputDimension, 1);   // Z = W^T * X + b
        A = MatrixXd::Zero(outputDimension, 1);   // A = activation(Z)

        dW = MatrixXd::Zero(inputDimension, outputDimension);
        db = MatrixXd::Zero(1, outputDimension);
    }

    MatrixXd Layer::forward(const MatrixXd &X) {
        const long N = X.rows();
        this->Z = MatrixXd::Zero(N, outputDimension);
        this->A = MatrixXd::Zero(N, outputDimension);

        const MatrixXd B = b.replicate(N, 1);
        this->Z = X * W + B;
        this->A = (*activation)(Z);
        return this->A;
    }
}   // namespace nn