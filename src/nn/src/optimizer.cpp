//
// Created by VanHoai on 9/6/25.
//

#include "nn/optimizer.hpp"

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

    void SGD::update(Layer &layer, MatrixXd &dW, MatrixXd &db) {
        const string layerId = getLayerId(layer);

        if (vW.find(layerId) == vW.end() || vB.find(layerId) == vB.end()) {
            vW[layerId] = MatrixXd::Zero(dW.rows(), dW.cols());
            vB[layerId] = MatrixXd::Zero(db.rows(), db.cols());
        }

        // Apply L2 regularization
        if (weight_decay > 0.0) {
            dW += weight_decay * layer.getW();
        }

        // Use reference to velocity
        MatrixXd vW_current = vW[layerId];
        MatrixXd vB_current = vB[layerId];

        const double eta = getLearningRate();

        if (nesterov) {
            vW_current = momentum * vW_current - eta * dW;
            vB_current = momentum * vB_current - eta * db;

            layer.setW(layer.getW() + momentum * vW_current - eta * dW);
            layer.setB(layer.getB() + momentum * vB_current - eta * db);
        } else {
            vW_current = momentum * vW_current - eta * dW;
            vB_current = momentum * vB_current - eta * db;

            layer.setW(layer.getW() + vW_current);
            layer.setB(layer.getB() + vB_current);
        }

        // Save updated velocities
        vW[layerId] = vW_current;
        vB[layerId] = vB_current;
    }

}   // namespace nn