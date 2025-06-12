//
// Created by VanHoai on 9/6/25.
//

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "layer.hpp"
#include <map>
#include <string>

using namespace std;

namespace nn {

    class Optimizer {
        private:
            double learning_rate;

        public:
            virtual ~Optimizer() = default;
            explicit Optimizer(const double learning_rate)
                : learning_rate(learning_rate) {}
            virtual void update(Layer &layer, MatrixXd &dW, MatrixXd &db);

            [[nodiscard]] double getLearningRate() const {
                return learning_rate;
            }

            void setLearningRate(const double learning_rate) {
                this->learning_rate = learning_rate;
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
            double weight_decay;

            map<string, MatrixXd> vW;
            map<string, MatrixXd> vB;

            static string getLayerId(Layer &layer) {
                const auto address = reinterpret_cast<uintptr_t>(&layer);
                return to_string(address);
            }

        public:
            explicit SGD(const double learning_rate,
                         const double momentum = 0.9,
                         const bool nesterov = true,
                         const double weight_decay = 0.0)
                : Optimizer(learning_rate), momentum(momentum),
                  nesterov(nesterov), weight_decay(weight_decay) {}

            void update(Layer &layer, MatrixXd &dW, MatrixXd &db) override;
    };

}   // namespace training

#endif   // OPTIMIZER_HPP
