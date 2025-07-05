//
// Created by VanHoai on 9/6/25.
//

#include "nn/optimizer.hpp"

namespace nn {
void Optimizer::update(Layer &layer, Eigen::MatrixXd &dW, Eigen::MatrixXd &db) {
    // W -= eta * dW
    // b -= eta * db

    Eigen::MatrixXd W = layer.getW();
    Eigen::MatrixXd b = layer.getB();

    W -= learningRate * dW;
    b -= learningRate * db;

    layer.setW(W);
    layer.setB(b);
}

void SGD::update(Layer &layer, Eigen::MatrixXd &dW, Eigen::MatrixXd &db) {
    const std::string layerId = getLayerId(layer);

    if (vW.find(layerId) == vW.end() || vb.find(layerId) == vb.end()) {
        vW[layerId] = Eigen::MatrixXd::Zero(dW.rows(), dW.cols());
        vb[layerId] = Eigen::MatrixXd::Zero(db.rows(), db.cols());
    }

    // Apply L2 regularization
    if (weightDecay > 0.0) {
        dW += weightDecay * layer.getW();
    }

    // Use reference to velocity
    Eigen::MatrixXd &vW = this->vW[layerId];
    Eigen::MatrixXd &vb = this->vb[layerId];

    const double eta = getLearningRate();

    if (nesterov) {
        vW = momentum * vW - eta * dW;
        vb = momentum * vb - eta * db;

        layer.setW(layer.getW() + momentum * vW - eta * dW);
        layer.setB(layer.getB() + momentum * vb - eta * db);
    } else {
        vW = momentum * vW - eta * dW;
        vb = momentum * vb - eta * db;

        layer.setW(layer.getW() + vW);
        layer.setB(layer.getB() + vb);
    }
}

void AdaGrad::update(Layer &layer, Eigen::MatrixXd &dW, Eigen::MatrixXd &db) {
    const std::string layerId = getLayerId(layer);

    // Initialize if not exists
    if (vW.find(layerId) == vW.end() || vb.find(layerId) == vb.end()) {
        vW[layerId] = Eigen::MatrixXd::Zero(dW.rows(), dW.cols());
        vb[layerId] = Eigen::MatrixXd::Zero(db.rows(), db.cols());
    }

    // Use references to internal velocity matrices
    Eigen::MatrixXd &vW = this->vW[layerId];
    Eigen::MatrixXd &vb = this->vb[layerId];

    // Accumulate squared gradients
    vW += dW.array().square().matrix();
    vb += db.array().square().matrix();

    // Compute adaptive updates
    const double eta = getLearningRate();
    const Eigen::MatrixXd newW = -eta * dW.array() / (vW.array().sqrt() + epsilon);
    const Eigen::MatrixXd newb = -eta * db.array() / (vb.array().sqrt() + epsilon);

    // Update parameters
    layer.setW(layer.getW() + newW);
    layer.setB(layer.getB() + newb);
}

void RMSProp::update(Layer &layer, Eigen::MatrixXd &dW, Eigen::MatrixXd &db) {
    const std::string layerId = getLayerId(layer);

    // Initialize if not exists
    if (vW.find(layerId) == vW.end() || vb.find(layerId) == vb.end()) {
        vW[layerId] = Eigen::MatrixXd::Zero(dW.rows(), dW.cols());
        vb[layerId] = Eigen::MatrixXd::Zero(db.rows(), db.cols());
    }

    // Use references to internal velocity matrices
    Eigen::MatrixXd &vW = this->vW[layerId];
    Eigen::MatrixXd &vb = this->vb[layerId];

    // Update moving average of squared gradients
    vW = beta * vW.array() + (1 - beta) * dW.array().square();
    vb = beta * vb.array() + (1 - beta) * db.array().square();

    // Apply parameter update with element-wise division
    const double eta = getLearningRate();
    const Eigen::MatrixXd newW = -eta * dW.array() / (vW.array().sqrt() + epsilon);
    const Eigen::MatrixXd newb = -eta * db.array() / (vb.array().sqrt() + epsilon);

    // Update parameters
    layer.setW(layer.getW() + newW);
    layer.setB(layer.getB() + newb);
}

void Adam::update(Layer &layer, Eigen::MatrixXd &dW, Eigen::MatrixXd &db) {
    const std::string layerId = getLayerId(layer);

    // Initialize if not exists
    if (vW.find(layerId) == vW.end()) {
        mW[layerId] = Eigen::MatrixXd::Zero(dW.rows(), dW.cols());
        mb[layerId] = Eigen::MatrixXd::Zero(db.rows(), db.cols());

        vW[layerId] = Eigen::MatrixXd::Zero(dW.rows(), dW.cols());
        vb[layerId] = Eigen::MatrixXd::Zero(db.rows(), db.cols());
    }

    // Update time step
    this->timeStep++;

    // Use references to internal moment and RMS matrices
    Eigen::MatrixXd &mW = this->mW[layerId];
    Eigen::MatrixXd &mb = this->mb[layerId];
    Eigen::MatrixXd &vW = this->vW[layerId];
    Eigen::MatrixXd &vb = this->vb[layerId];

    // Update moment estimates (m)
    mW = beta1 * mW.array() + (1 - beta1) * dW.array();
    mb = beta1 * mb.array() + (1 - beta1) * db.array();

    // Update RMS estimates (v)
    vW = beta2 * vW.array() + (1 - beta2) * dW.array().square();
    vb = beta2 * vb.array() + (1 - beta2) * db.array().square();

    // Bias correction
    const auto t = static_cast<double>(this->timeStep);
    Eigen::MatrixXd mW_hat = mW.array() / (1 - std::pow(beta1, t));
    Eigen::MatrixXd vW_hat = vW.array() / (1 - std::pow(beta2, t));

    Eigen::MatrixXd mb_hat = mb.array() / (1 - std::pow(beta1, t));
    Eigen::MatrixXd vb_hat = vb.array() / (1 - std::pow(beta2, t));

    // Update weights and biases
    const double eta = getLearningRate();
    const Eigen::MatrixXd newW = -eta * mW_hat.array() / (vW_hat.array().sqrt() + epsilon);
    const Eigen::MatrixXd newb = -eta * mb_hat.array() / (vb_hat.array().sqrt() + epsilon);

    layer.setW(layer.getW() + newW);
    layer.setB(layer.getB() + newb);
}

}  // namespace nn