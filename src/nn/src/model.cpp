//
// Created by VanHoai on 9/6/25.
//

#include "nn/model.hpp"

#include <iostream>

#include "core/multivariate_normal.hpp"

namespace nn {
    Sequential::Sequential(std::vector<std::unique_ptr<Layer>> &layers, std::unique_ptr<Loss> &loss,
                           std::unique_ptr<Optimizer> &optimizer) {
        this->layers = std::move(layers);
        this->loss = std::move(loss);
        this->optimizer = std::move(optimizer);

        this->N = 1000;
        this->batchSize = 32;

        this->input = Eigen::MatrixXd();
        this->output = Eigen::MatrixXd();
    }

    Eigen::MatrixXd Sequential::feedforward(Eigen::MatrixXd &X) {
        this->input = X;

        Eigen::MatrixXd O = X;
        for (const auto &layer : layers) O = layer->forward(O);

        this->output = O;
        return O;
    }

    void Sequential::backpropagation(Eigen::MatrixXd &Y) {
        Eigen::MatrixXd A = this->output;

        Eigen::MatrixXd dA = this->loss->derivative(Y, A);
        Eigen::MatrixXd dZ = Eigen::MatrixXd::Zero(dA.rows(), dA.cols());

        for (int idx = layers.size() - 1; idx >= 0; --idx) {
            const std::unique_ptr<Layer> &layer = layers[idx];
            dZ = dA.array() * layer->derivativeActivation().array();

            Eigen::MatrixXd AP;
            if (idx > 0)
                AP = layers[idx - 1]->getA();
            else
                AP = input;

            Eigen::MatrixXd dW = AP.transpose() * dZ / batchSize;
            Eigen::MatrixXd db = dZ.colwise().sum() / batchSize;

            layer->setDW(dW);
            layer->setDb(db);

            if (idx > 0)
                dA = dZ * layer->getW().transpose();
        }
    }

    void Sequential::update() {
        for (const auto &layer : layers) {
            Eigen::MatrixXd dW = layer->getDW();
            Eigen::MatrixXd db = layer->getDb();

            optimizer->update((*layer), dW, db);
        }
    }

    void Sequential::fit(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, const int epochs,
                         const int batchSize, const bool verbose, const int frequency,
                         std::optional<EarlyStopping> earlyStopping) {
        const int N = static_cast<int>(X.rows());
        this->batchSize = batchSize;

        core::MultivariateNormal mvn;
        Eigen::MatrixXd shuffleX = Eigen::MatrixXd::Zero(X.rows(), X.cols());
        Eigen::MatrixXd shuffleY = Eigen::MatrixXd::Zero(Y.rows(), Y.cols());

        for (int epoch = 0; epoch < epochs; ++epoch) {
            std::vector<int> indices = mvn.shuffleIndices(N);
            for (int i = 0; i < N; ++i) {
                shuffleX.row(i) = X.row(indices[i]);
                shuffleY.row(i) = Y.row(indices[i]);
            }

            for (int start = 0; start < N; start += batchSize) {
                const int end = std::min(start + batchSize, N);
                Eigen::MatrixXd batchX = shuffleX.block(start, 0, end - start, X.cols());
                Eigen::MatrixXd batchY = shuffleY.block(start, 0, end - start, Y.cols());
                this->feedforward(batchX);
                this->backpropagation(batchY);
                this->update();
            }

            if (verbose && epoch % frequency == 0) {
                Eigen::MatrixXd A = this->predict(X);
                const double loss = this->loss->operator()(Y, A);
                const double accuracy = this->evaluate(Y, A);
                std::cout << "Epoch: " << epoch << ", Loss: " << loss << ", Accuracy: " << accuracy
                          << std::endl;

                if (earlyStopping.has_value()) {
                    const int numLayer = static_cast<int>(layers.size());
                    std::vector<Eigen::MatrixXd> Ws(numLayer);
                    std::vector<Eigen::MatrixXd> bs(numLayer);

                    if (earlyStopping->on_epoch_end(epoch, accuracy, Ws, bs)) {
                        std::cout << "Early stopping at epoch: " << epoch << std::endl;
                        if (earlyStopping->getIsStore()) {
                            // Update best weights and biases
                            for (int i = 0; i < numLayer; i++) {
                                layers[i]->setDW(Ws[i]);
                                layers[i]->setDb(bs[i]);
                            }
                        }

                        return;
                    }
                }
            }
        }
    }

    Eigen::MatrixXd Sequential::predict(Eigen::MatrixXd &X) {
        Eigen::MatrixXd O = this->feedforward(X);
        return O;
    }

    double Sequential::calculateLoss(Eigen::MatrixXd &Y, Eigen::MatrixXd &A) {
        const double loss = this->loss->operator()(Y, A);
        return loss;
    }

    double Sequential::evaluate(Eigen::MatrixXd &Y, Eigen::MatrixXd &A) {
        const long N = Y.rows();
        int correct = 0;
        for (int i = 0; i < N; i++) {
            int yIndex, aIndex;
            Y.row(i).maxCoeff(&yIndex);
            A.row(i).maxCoeff(&aIndex);

            if (yIndex == aIndex)
                correct++;
        }

        return static_cast<double>(correct) / N;
    }

    void Sequential::load(const std::string &path) {}
    void Sequential::save(const std::string &path) {}

}   // namespace nn