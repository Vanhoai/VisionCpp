//
// Created by VanHoai on 9/6/25.
//

#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Core>

#include "activation.hpp"

namespace nn {

    class Layer {
        private:
            // di & do for the layer
            int inputDimension;
            int outputDimension;

            // activation function like: ReLU, Sigmoid, Softmax
            std::unique_ptr<Activation> activation;

            // weights and biases
            Eigen::MatrixXd W, b;

            // props for backpropagation
            Eigen::MatrixXd Z, A;
            Eigen::MatrixXd dW, db;

        public:
            virtual ~Layer() = default;
            Layer(int inputDimension, int outputDimension, std::unique_ptr<Activation> activation);

            virtual Eigen::MatrixXd forward(const Eigen::MatrixXd &X);

            [[nodiscard]] Eigen::MatrixXd derivativeActivation() const {
                Eigen::MatrixXd ZCopy = Z;
                return activation->derivative(ZCopy);
            }

            [[nodiscard]] Eigen::MatrixXd getW() const { return W; }
            void setW(const Eigen::MatrixXd &W) { this->W = W; }

            [[nodiscard]] Eigen::MatrixXd getB() const { return b; }
            void setB(const Eigen::MatrixXd &b) { this->b = b; }

            [[nodiscard]] Eigen::MatrixXd getA() const { return A; }
            void setA(const Eigen::MatrixXd &A) { this->A = A; }

            [[nodiscard]] Eigen::MatrixXd getDW() const { return dW; }
            void setDW(const Eigen::MatrixXd &dW) { this->dW = dW; }

            [[nodiscard]] Eigen::MatrixXd getDb() const { return db; }
            void setDb(const Eigen::MatrixXd &db) { this->db = db; }

            [[nodiscard]] int getInputDimension() const { return inputDimension; }

            [[nodiscard]] int getOutputDimension() const { return outputDimension; }

            [[nodiscard]] virtual std::string getName() const = 0;
    };

    class ReLU final : public Layer {
        public:
            ReLU(const int inputDimension, const int outputDimension)
                : Layer(inputDimension, outputDimension, std::make_unique<ReLUActivation>()) {}

            [[nodiscard]] std::string getName() const override { return "ReLU"; }
    };

    class Sigmoid final : public Layer {
        public:
            Sigmoid(const int inputDimension, const int outputDimension)
                : Layer(inputDimension, outputDimension, std::make_unique<SigmoidActivation>()) {}

            [[nodiscard]] std::string getName() const override { return "Sigmoid"; }
    };

    class Softmax final : public Layer {
        public:
            Softmax(const int inputDimension, const int outputDimension)
                : Layer(inputDimension, outputDimension, std::make_unique<SoftmaxActivation>()) {}

            [[nodiscard]] std::string getName() const override { return "Softmax"; }
    };

}   // namespace nn

#endif   // LAYER_HPP
