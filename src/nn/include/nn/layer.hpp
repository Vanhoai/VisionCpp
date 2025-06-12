//
// Created by VanHoai on 9/6/25.
//

#ifndef LAYER_HPP
#define LAYER_HPP

#include "activation.hpp"
#include <Eigen/Core>

using namespace std;
using namespace Eigen;

namespace nn {

    class Layer {
        private:
            // di & do for the layer
            int inputDimension;
            int outputDimension;

            // activation function like: ReLU, Sigmoid, Softmax
            unique_ptr<Activation> activation;

            // weights and biases
            MatrixXd W, b;

            // props for backpropagation
            MatrixXd Z, A;
            MatrixXd dW, db;

        public:
            virtual ~Layer() = default;
            Layer(int inputDimension, int outputDimension,
                  unique_ptr<Activation> activation);

            virtual MatrixXd forward(const MatrixXd &X);

            [[nodiscard]] MatrixXd derivativeActivation() const {
                MatrixXd ZCopy = Z;
                return activation->derivative(ZCopy);
            }

            [[nodiscard]] MatrixXd getW() const { return W; }
            void setW(const MatrixXd &W) { this->W = W; }

            [[nodiscard]] MatrixXd getB() const { return b; }
            void setB(const MatrixXd &b) { this->b = b; }

            [[nodiscard]] MatrixXd getA() const { return A; }
            void setA(const MatrixXd &A) { this->A = A; }

            [[nodiscard]] MatrixXd getDW() const { return dW; }
            void setDW(const MatrixXd &dW) { this->dW = dW; }

            [[nodiscard]] MatrixXd getDb() const { return db; }
            void setDb(const MatrixXd &db) { this->db = db; }

            [[nodiscard]] int getInputDimension() const {
                return inputDimension;
            }

            [[nodiscard]] int getOutputDimension() const {
                return outputDimension;
            }

            [[nodiscard]] virtual string getName() const = 0;
    };

    class ReLU final : public Layer {
        public:
            ReLU(const int inputDimension, const int outputDimension)
                : Layer(inputDimension, outputDimension,
                        make_unique<ReLUActivation>()) {}

            [[nodiscard]] string getName() const override { return "ReLU"; }
    };

    class Sigmoid final : public Layer {
        public:
            Sigmoid(const int inputDimension, const int outputDimension)
                : Layer(inputDimension, outputDimension,
                        make_unique<SigmoidActivation>()) {}

            [[nodiscard]] string getName() const override { return "Sigmoid"; }
    };

    class Softmax final : public Layer {
        public:
            Softmax(const int inputDimension, const int outputDimension)
                : Layer(inputDimension, outputDimension,
                        make_unique<SoftmaxActivation>()) {}

            [[nodiscard]] string getName() const override { return "Softmax"; }
    };

}   // namespace nn

#endif   // LAYER_HPP
