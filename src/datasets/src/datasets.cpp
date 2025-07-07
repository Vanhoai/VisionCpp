//
// Created by VanHoai on 11/6/25.
//

#include "datasets/datasets.hpp"

#include <iostream>
#include <vector>

#include "core/multivariate_normal.hpp"

namespace datasets {

bool Dataset::load(Eigen::MatrixXd& XTrain, Eigen::MatrixXd& YTrain, Eigen::MatrixXd& XTest,
                   Eigen::MatrixXd& YTest, const int percentTrain, const bool isShuffle) const {
    if (!getIsSetup()) {
        std::cerr << "Dataset is not setup yet. Please call setup() before load()." << std::endl;
        return false;
    }

    // ensure the provided data is valid
    assert(X.rows() == N && X.cols() == classes);
    assert(Y.size() == N);
    assert(X.rows() == Y.size());

    Eigen::MatrixXd FX = X;  // R(N, d)
    Eigen::VectorXd FY = Y;  // R(N, 1)

    if (isShuffle) {
        core::MultivariateNormal mvn;
        const std::vector<int> indices = mvn.shuffleIndices(N);

        for (int i = 0; i < N; ++i) {
            FX.row(i) = X.row(indices[i]);
            FY(i) = Y(indices[i]);
        }
    }

    const int numTrain = static_cast<int>(percentTrain / 100.0 * N);
    const int numTest = N - numTrain;

    XTrain = FX.topRows(numTrain);
    XTest = FX.bottomRows(numTest);

    YTrain = Eigen::MatrixXd::Zero(numTrain, classes);
    YTest = Eigen::MatrixXd::Zero(numTest, classes);

    for (int i = 0; i < numTrain; ++i) YTrain(i, static_cast<int>(FY(i))) = 1.0;
    for (int i = 0; i < numTest; ++i) YTest(i, static_cast<int>(FX(numTrain + i))) = 1.0;

    return true;
}

bool Dataset::load(Eigen::MatrixXd& XTrain, Eigen::VectorXd& YTrain, Eigen::MatrixXd& XTest,
                   Eigen::VectorXd& YTest, const int percentTrain, const bool isShuffle) const {
    if (!getIsSetup()) {
        std::cerr << "Dataset is not setup yet. Please call setup() before load()." << std::endl;
        return false;
    }

    // ensure the provided data is valid
    assert(X.rows() == N && X.cols() == classes);
    assert(Y.size() == N);
    assert(X.rows() == Y.size());

    Eigen::MatrixXd FX = X;  // R(N, d)
    Eigen::VectorXd FY = Y;  // R(N, 1)

    if (isShuffle) {
        core::MultivariateNormal mvn;
        const std::vector<int> indices = mvn.shuffleIndices(N);

        for (int i = 0; i < N; ++i) {
            FX.row(i) = X.row(indices[i]);
            FY(i) = Y(indices[i]);
        }
    }

    const int numTrain = static_cast<int>(percentTrain / 100.0 * N);
    XTrain = FX.topRows(numTrain);
    XTest = FX.bottomRows(N - numTrain);
    YTrain = FY.topRows(numTrain);
    YTest = FY.bottomRows(N - numTrain);

    return true;
}

void TwoDimensionDataset::setup(const Eigen::MatrixXd& mean, const Eigen::MatrixXd& covariance) {
    const int N = this->getN();
    const int classes = this->getClasses();
    constexpr int d = 2;

    assert(mean.rows() == N);
    assert(mean.cols() == d);  // datapoint has 2 dimensions

    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(N, d);
    Eigen::VectorXd Y = Eigen::VectorXd::Zero(N);

    core::MultivariateNormal mvn;
    const int P = N / classes;

    for (int i = 0; i < classes; ++i) {
        const Eigen::MatrixXd Xi = mvn.random(P, mean.row(i), covariance);
        // Mark X[i] to block (i * P to (i + 1) * P)
        X.block(i * P, 0, P, d) = Xi;

        // Mark label y[i] to block (i * P to (i + 1) * P)
        for (int k = 0; k < P; ++k) Y(i * P + k) = i;
    }

    this->setX(X);
    this->setY(Y);
    this->setIsSetup(true);
}

}  // namespace datasets
