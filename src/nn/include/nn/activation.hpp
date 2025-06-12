//
// Created by VanHoai on 8/6/25.
//

#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <Eigen/Core>
using namespace Eigen;

namespace nn {

    class Activation {
        public:
            virtual ~Activation() = default;
            virtual MatrixXd operator()(MatrixXd &X) = 0;
            virtual MatrixXd derivative(MatrixXd &X) = 0;
    };

    class ReLUActivation final : public Activation {
        public:
            /**
             * ReLU activation function
             * @param X: input matrix
             * @return: output matrix with ReLU applied
             *
             * @formula: f(x) = max(0, x)
             */
            MatrixXd operator()(MatrixXd &X) override;

            /**
             * Derivative of ReLU activation function
             * @param X: input matrix
             * @return: output matrix with derivative applied
             *
             * @formula: f'(x) = 1 if x > 0, else 0
             */
            MatrixXd derivative(MatrixXd &X) override;
    };

    class SigmoidActivation final : public Activation {
        public:
            /**
             * Sigmoid activation function
             * @param X: input matrix
             * @return: output matrix with Sigmoid applied
             *
             * @formula: f(x) = 1 / (1 + exp(-x))
             */
            MatrixXd operator()(MatrixXd &X) override;

            /**
             * Derivative of Sigmoid activation function
             * @param X: input matrix
             * @return: output matrix with derivative applied
             *
             * @formula: f'(x) = f(x) * (1 - f(x))
             */
            MatrixXd derivative(MatrixXd &X) override;
    };

    class SoftmaxActivation final : public Activation {
        public:
            /**
             * Softmax activation function
             * @param X: input matrix
             * @return: output matrix with Softmax applied
             *
             * @formula: f(x) = exp(x) / Σ(exp(x))
             */
            MatrixXd operator()(MatrixXd &X) override;

            /**
             * Derivative of Softmax activation function
             * @param X: input matrix
             * @return: output matrix with derivative applied
             *
             * @notice: the derivative is not straightforward, it is usually
             * computed during backpropagation.
             */
            MatrixXd derivative(MatrixXd &X) override;
    };

}   // namespace training

#endif   // ACTIVATION_HPP
