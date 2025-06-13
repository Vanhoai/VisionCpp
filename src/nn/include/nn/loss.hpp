//
// Created by VanHoai on 8/6/25.
//

#ifndef LOSS_HPP
#define LOSS_HPP

#include <Eigen/Core>

namespace nn {
    class Loss {
        public:
            virtual ~Loss() = default;
            virtual double operator()(Eigen::MatrixXd &Y, Eigen::MatrixXd &A) = 0;
            virtual Eigen::MatrixXd derivative(Eigen::MatrixXd &Y, Eigen::MatrixXd &A) = 0;
    };

    class CrossEntropyLoss final : public Loss {
        private:
            double epsilon = 1e-15;

        public:
            CrossEntropyLoss() = default;
            /**
             * Cross-entropy loss function
             * @param Y: true labels (one-hot encoded)
             * @param A: predicted probabilities (one-hot encoded)
             * @return: cross-entropy loss value
             *
             * @formula: L(Y, A) = -Σ(Y * log(A))
             */
            double operator()(Eigen::MatrixXd &Y, Eigen::MatrixXd &A) override;

            /**
             * Derivative of the cross-entropy loss function
             * @param Y: true labels (one-hot encoded)
             * @param A: predicted probabilities (one-hot encoded)
             * @return: derivative of the loss with respect to A
             *
             * @notice: if activation function is softmax, this function
             * should return A - Y to simplify the backpropagation
             */
            Eigen::MatrixXd derivative(Eigen::MatrixXd &Y, Eigen::MatrixXd &A) override;
    };
}   // namespace nn

#endif   // LOSS_HPP
