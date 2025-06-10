//
// Created by VanHoai on 9/6/25.
//

#include "../optimizer/optimizer.hpp"

namespace nn {
    void Optimizer::update(Layer &layer, MatrixXd &dW, MatrixXd &db) {
        // W -= eta * dW
        // b -= eta * db

        layer.setW(layer.getW() - learning_rate * dW);
        layer.setB(layer.getB() - learning_rate * db);
    }

}   // namespace nn