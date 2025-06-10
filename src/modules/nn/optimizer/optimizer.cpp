//
// Created by VanHoai on 9/6/25.
//

#include "../optimizer/optimizer.hpp"

namespace nn {
    void Optimizer::update(Layer &layer, MatrixXd &dW, MatrixXd &db) {
        // W -= eta * dW
        // b -= eta * db

        MatrixXd W = layer.getW();
        MatrixXd b = layer.getB();

        W -= learning_rate * dW;
        b -= learning_rate * db;

        layer.setW(W);
        layer.setB(b);
    }

}   // namespace nn