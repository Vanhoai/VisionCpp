//
// Created by VanHoai on 8/6/25.
//

#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <Eigen/Core>

namespace nn {

class Activation {
public:
    virtual ~Activation() = default;
    virtual Eigen::MatrixXd operator()(Eigen::MatrixXd &X) = 0;
    virtual Eigen::MatrixXd derivative(Eigen::MatrixXd &X) = 0;
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
    Eigen::MatrixXd operator()(Eigen::MatrixXd &X) override;

    /**
     * Derivative of ReLU activation function
     * @param X: input matrix
     * @return: output matrix with derivative applied
     *
     * @formula: f'(x) = 1 if x > 0, else 0
     */
    Eigen::MatrixXd derivative(Eigen::MatrixXd &X) override;
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
    Eigen::MatrixXd operator()(Eigen::MatrixXd &X) override;

    /**
     * Derivative of Sigmoid activation function
     * @param X: input matrix
     * @return: output matrix with derivative applied
     *
     * @formula: f'(x) = f(x) * (1 - f(x))
     */
    Eigen::MatrixXd derivative(Eigen::MatrixXd &X) override;
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
    Eigen::MatrixXd operator()(Eigen::MatrixXd &X) override;

    /**
     * Derivative of Softmax activation function
     * @param X: input matrix
     * @return: output matrix with derivative applied
     *
     * @notice: the derivative is not straightforward, it is usually
     * computed during backpropagation.
     */
    Eigen::MatrixXd derivative(Eigen::MatrixXd &X) override;
};

}  // namespace nn

#endif  // ACTIVATION_HPP
