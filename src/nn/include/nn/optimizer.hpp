//
// Created by VanHoai on 9/6/25.
//

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <map>
#include <string>

#include "layer.hpp"

namespace nn {

    class Optimizer {
        private:
            double learningRate;

        public:
            virtual ~Optimizer() = default;
            explicit Optimizer(const double learningRate) : learningRate(learningRate) {}
            virtual void update(Layer &layer, Eigen::MatrixXd &dW, Eigen::MatrixXd &db);

            [[nodiscard]] double getLearningRate() const { return learningRate; }
            void setLearningRate(const double learningRate) { this->learningRate = learningRate; }

            static std::string getLayerId(Layer &layer) {
                const auto address = reinterpret_cast<uintptr_t>(&layer);
                return std::to_string(address);
            }
    };

    /**
     * Stochastic Gradient Descent (SGD)
     * This optimizer updates the parameters using the gradient of the
     * loss function. Formula: w = w - η * g where:
     * - w is the parameter (weights or biases)
     * - η is the learning rate
     * - g is the gradient of the loss function with respect to the
     * parameter
     *
     * Parameters:
     * - eta: Learning rate (default: 1e-3)
     * - momentum: Momentum factor (default: 0.9)
     * - nesterov: Whether to use Nesterov Accelerated Gradient
     * (default: True)
     * - weight_decay: L2 regularization factor (default: 0.0)
     *
     * if you use momentum formula becomes:
     * w = w + momentum * v - η * g
     * where:
     * - v is the velocity (previous update)
     * - momentum is the momentum factor
     * - nesterov: If True, uses Nesterov Accelerated Gradient (NAG)
     * which looks ahead at the next position
     * - weight_decay: If non-zero, applies L2 regularization to the
     * weights
     */
    class SGD final : public Optimizer {
        private:
            double momentum;
            bool nesterov;
            double weightDecay;

            std::map<std::string, Eigen::MatrixXd> vW;
            std::map<std::string, Eigen::MatrixXd> vb;

        public:
            explicit SGD(const double learningRate, const double momentum = 0.9,
                         const bool nesterov = true, const double weight_decay = 0.0)
                : Optimizer(learningRate),
                  momentum(momentum),
                  nesterov(nesterov),
                  weightDecay(weight_decay) {}

            void update(Layer &layer, Eigen::MatrixXd &dW, Eigen::MatrixXd &db) override;
    };

    /**
     * Adaptive Gradient Algorithm (AdaGrad)
     * This optimizer adapts the learning rate for each parameter based on the historical gradients.
     * It is particularly useful for dealing with sparse data and features.
     * Formula:
     * vt = vt + g^2
     * w = w - η / (sqrt(vt) + ε)  * g
     * where:
     * - vt is the accumulated squared gradient
     * - g is the gradient
     * - η is the learning rate
     * - ε is a small constant to avoid division by zero
     */
    class AdaGrad final : public Optimizer {
        private:
            double epsilon;

            std::map<std::string, Eigen::MatrixXd> vW;
            std::map<std::string, Eigen::MatrixXd> vb;

        public:
            explicit AdaGrad(const double learningRate, const double epsilon = 1e-15)
                : Optimizer(learningRate), epsilon(epsilon) {}

            void update(Layer &layer, Eigen::MatrixXd &dW, Eigen::MatrixXd &db) override;
    };

    /**
     *  Root Mean Square Propagation (RMSProp)
     * This optimizer is designed to adapt the learning rate for each parameter based on the average
     * of recent gradients. It helps to stabilize the learning process and is particularly effective
     * for non-stationary objectives. Formula: vt = β * vt-1 + (1 - β) * g^2 w = w - η / (sqrt(vt) +
     * ε) * g where:
     * - vt is the moving average of squared gradients
     * - β is the decay rate (typically around 0.9 or 0.95)
     * - g is the gradient
     * - η is the learning rate
     * - ε is a small constant to avoid division by zero
     */
    class RMSProp final : public Optimizer {
        private:
            double epsilon;
            double beta;

            std::map<std::string, Eigen::MatrixXd> vW;
            std::map<std::string, Eigen::MatrixXd> vb;

        public:
            explicit RMSProp(const double learningRate, const double epsilon = 1e-15,
                             const double beta = 0.9)
                : Optimizer(learningRate), epsilon(epsilon), beta(beta) {}

            void update(Layer &layer, Eigen::MatrixXd &dW, Eigen::MatrixXd &db) override;
    };

    /**
     *  Adam (Adaptive Moment Estimation)
     * This optimizer combines the benefits of momentum and RMSProp.
     * It maintains both the first moment (mean) and the second moment (variance) of the gradients.
     * It is widely used due to its efficiency and effectiveness in training deep neural networks.
     * Formula:
     * m = β1 * m + (1 - β1) * g
     * v = β2 * v + (1 - β2) * g^2
     * m_hat = m / (1 - β1^t)
     * v_hat = v / (1 - β2^t)
     * w = w - η * m_hat / (sqrt(v_hat) + ε)
     * where:
     * - m is the first moment (mean of exponential moving average of gradients)
     * - v is the second moment (mean of exponential moving average of squared gradients)
     * - β1 is the decay rate for the first moment (typically around 0.9)
     * - β2 is the decay rate for the second moment (typically around 0.999)
     * - m_hat and v_hat are bias-corrected estimates of the first and second moments
     * - g is the gradient
     * - η is the learning rate
     * - ε is a small constant to avoid division by zero
     */
    class Adam final : public Optimizer {
        private:
            double epsilon;
            double beta1;
            double beta2;

            // mean of exponential moving average of gradients
            std::map<std::string, Eigen::MatrixXd> mW;
            std::map<std::string, Eigen::MatrixXd> mb;

            // mean of exponential moving average of squared gradients
            std::map<std::string, Eigen::MatrixXd> vW;
            std::map<std::string, Eigen::MatrixXd> vb;

            int timeStep = 0;   // Time step

        public:
            explicit Adam(const double learningRate, const double epsilon = 1e-15,
                          const double beta1 = 0.9, const double beta2 = 0.999)
                : Optimizer(learningRate), epsilon(epsilon), beta1(beta1), beta2(beta2) {}

            void update(Layer &layer, Eigen::MatrixXd &dW, Eigen::MatrixXd &db) override;
    };

}   // namespace nn

#endif   // OPTIMIZER_HPP
