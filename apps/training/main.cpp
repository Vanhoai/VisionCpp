//
// Created by VanHoai on 12/6/25.
//

#include <omp.h>

#include <chrono>
#include <iostream>

#include "datasets/datasets.hpp"
#include "nn/layer.hpp"
#include "nn/loss.hpp"
#include "nn/model.hpp"
#include "nn/optimizer.hpp"

int main() {
    Eigen::initParallel();
    Eigen::setNbThreads(4);

    const int threads = omp_get_num_threads();
    std::cout << "Number of threads available: " << threads << std::endl;
    std::cout << "Number threads use by Eigen: " << Eigen::nbThreads() << std::endl;

    constexpr int N = 100000;
    constexpr int d = 2;
    constexpr int classes = 8;

    constexpr bool isShuffle = true;
    Eigen::MatrixXd XTrain, XTest;
    Eigen::MatrixXd YTrain, YTest;
    Eigen::MatrixXd covariance(2, 2);
    covariance << 1, 0, 0, 1;
    Eigen::MatrixXd mean(classes, d);
    mean << 1, 1, 1, 6, 1, 12, 6, 1, 12, 6, 6, 6, 6, 12, 12, 12;

    datasets::TwoDimensionDataset dataset(N, classes);
    dataset.setup(mean, covariance);
    dataset.load(XTrain, YTrain, XTest, YTest, 80, isShuffle);

    int d1 = 10;

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

    constexpr int epochs = 100;
    constexpr int batchSize = 256;
    sequential.fit(XTrain, YTrain, epochs, batchSize, true, 10, std::nullopt);

    return EXIT_SUCCESS;
}
