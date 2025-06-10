//
// Created by VanHoai on 9/6/25.
//

#include "model.hpp"
#include <iostream>

namespace nn {
    Sequential::Sequential(vector<unique_ptr<nn::Layer>> &layers,
                           unique_ptr<Loss> &loss,
                           unique_ptr<Optimizer> &optimizer) {
        this->layers = std::move(layers);
        this->loss = std::move(loss);
        this->optimizer = std::move(optimizer);

        this->N = 1000;
        this->batchSize = 32;

        this->input = MatrixXd();
        this->output = MatrixXd();
    }

    MatrixXd Sequential::feedforward(MatrixXd &X) {
        this->input = X;

        MatrixXd O = X;
        for (const auto &layer : layers)
            O = layer->forward(O);

        this->output = O;
        return O;
    }

    void Sequential::backpropagation(MatrixXd &Y) {
        MatrixXd A = this->output;

        MatrixXd dA = this->loss->derivative(Y, A);
        MatrixXd dZ = MatrixXd::Zero(dA.rows(), dA.cols());

        for (int idx = layers.size() - 1; idx >= 0; --idx) {
            const unique_ptr<Layer> &layer = layers[idx];
            dZ = dA.array() * layer->derivativeActivation().array();

            MatrixXd AP;
            if (idx > 0)
                AP = layers[idx - 1]->getA();
            else
                AP = input;

            MatrixXd dW = AP.transpose() * dZ / batchSize;
            MatrixXd db = dZ.colwise().sum() / batchSize;

            layer->setDW(dW);
            layer->setDb(db);

            if (idx > 0)
                dA = dZ * layer->getW().transpose();
        }
    }

    void Sequential::update() {
        for (const auto &layer : layers) {
            MatrixXd dW = layer->getDW();
            MatrixXd db = layer->getDb();

            optimizer->update((*layer), dW, db);
        }
    }

    void Sequential::fit(MatrixXd &X, MatrixXd &Y, const int epochs,
                         const int batchSize) {
        const int N = static_cast<int>(X.rows());
        this->batchSize = batchSize;

        for (int epoch = 0; epoch < epochs; ++epoch) {

            for (int start = 0; start < N; start += batchSize) {
                const int end = min(start + batchSize, N);
                MatrixXd batchX = X.block(start, 0, end - start, X.cols());
                MatrixXd batchY = Y.block(start, 0, end - start, Y.cols());
                this->feedforward(batchX);
                this->backpropagation(batchY);
                this->update();
            }

            if (epoch % 100 == 0) {
                MatrixXd A = this->predict(X);
                const double loss = this->loss->operator()(Y, A);
                const double accuracy = this->evaluate(Y, A);
                cout << "Epoch: " << epoch << ", Loss: " << loss
                     << ", Accuracy: " << accuracy << endl;
            }
        }
    }

    MatrixXd Sequential::predict(MatrixXd &X) {
        MatrixXd O = this->feedforward(X);
        return O;
    }

    double Sequential::evaluate(MatrixXd &Y, MatrixXd &A) {
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

    void Sequential::load(const string &path) {}

    void Sequential::save(const string &path) {}

}   // namespace nn