//
// Created by VanHoai on 9/6/25.
//

#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "../layer/layer.hpp"

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

    class SGD final : public Optimizer {
        public:
            explicit SGD(const double learning_rate)
                : Optimizer(learning_rate) {}

            friend std::ostream &operator<<(std::ostream &os,
                                            const SGD &optimizer) {
                os << "SGD Optimizer with learning rate: "
                   << optimizer.getLearningRate();
                return os;
            }
    };

}   // namespace nn

#endif   // OPTIMIZER_HPP
