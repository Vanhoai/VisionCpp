//
// Created by VanHoai on 12/6/25.
//

#include <omp.h>

#include <chrono>
#include <iostream>

#include "nn/layer.hpp"
#include "nn/loss.hpp"
#include "nn/model.hpp"
#include "nn/optimizer.hpp"
#include "nn/prepare.hpp"

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);

    initParallel();
    setNbThreads(4);

    const int threads = omp_get_num_threads();
    std::cout << "Number of threads available: " << threads << std::endl;
    std::cout << "Number threads use by Eigen: " << nbThreads() << std::endl;

    constexpr int N = 1000000;
    int d = 2;
    int d1 = 5;
    int classes = 4;

    MatrixXd X(N, d);
    MatrixXd Y(N, classes);
    nn::prepare(N, classes, d, X, Y);

    constexpr int T = N * 0.8;
    MatrixXd XTrain = X.block(0, 0, T, d);
    MatrixXd YTrain = Y.block(0, 0, T, classes);
    MatrixXd XTest = X.block(T, 0, N - T, d);
    MatrixXd YTest = Y.block(T, 0, N - T, classes);

    std::vector<std::unique_ptr<nn::Layer>> layers;
    layers.push_back(std::make_unique<nn::ReLU>(d, d1));
    layers.push_back(std::make_unique<nn::Softmax>(d1, classes));

    // Hyperparameters
    double learningRate = 1e-3;
    double momentum = 0.9;
    bool nesterov = true;
    double weightDecay = 0.0;
    double epsilon = 1e-15;
    double beta = 0.9;
    double beta1 = 0.9;
    double beta2 = 0.999;

    std::unique_ptr<nn::Loss> loss = std::make_unique<nn::CrossEntropyLoss>();
    std::unique_ptr<nn::Optimizer> SGD =
        std::make_unique<nn::SGD>(learningRate, momentum, nesterov, weightDecay);

    std::unique_ptr<nn::Optimizer> AdaGrad = std::make_unique<nn::AdaGrad>(learningRate, epsilon);
    std::unique_ptr<nn::Optimizer> RMSProp =
        std::make_unique<nn::RMSProp>(learningRate, epsilon, beta);
    std::unique_ptr<nn::Optimizer> Adam =
        std::make_unique<nn::Adam>(learningRate, epsilon, beta1, beta2);

    nn::Sequential sequential(layers, loss, Adam);
    constexpr int epochs = 5;
    constexpr int batchSize = 256;
    constexpr bool verbose = true;
    constexpr int frequency = 1;

    const nn::EarlyStopping earlyStopping(10, 1e-3, true, nn::MonitorEarlyStopping::ValidationLoss);
    sequential.fit(XTrain, YTrain, epochs, batchSize, verbose, frequency, earlyStopping);

    MatrixXd O = sequential.predict(XTest);
    const double lossValue = sequential.calculateLoss(YTest, O);
    const double accuracyValue = sequential.evaluate(YTest, O);
    std::cout << "Final Loss: " << lossValue << ", Final Accuracy: " << accuracyValue << std::endl;

    return EXIT_SUCCESS;
}
