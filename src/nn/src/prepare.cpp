//
// Created by VanHoai on 10/6/25.
//

#include <core/multivariate_normal.hpp>
#include <nn/prepare.hpp>

namespace nn {
    void prepare(const int N, const int groups, const int d, Eigen::MatrixXd &X,
                 Eigen::MatrixXd &Y) {
        X = Eigen::MatrixXd::Zero(N, d);
        Y = Eigen::MatrixXd::Zero(N, groups);

        Eigen::MatrixXd means(4, 2);
        means << 1, 1, 1, 6, 6, 1, 6, 6;

        Eigen::MatrixXd covariance(2, 2);
        covariance << 1, 0, 0, 1;

        const int P = N / groups;

        core::MultivariateNormal mvn;

        Eigen::VectorXd y(N);
        Eigen::MatrixXd shuffledX(X.rows(), X.cols());

        for (int i = 0; i < groups; ++i) {
            const Eigen::MatrixXd Xi = mvn.random(P, means.row(i), covariance);
            shuffledX.block(i * P, 0, P, d) = Xi;
            for (int j = 0; j < P; ++j) y(i * P + j) = i;
        }

        const std::vector<int> indices = mvn.shuffleIndices(N);

        Eigen::VectorXd shuffledY(y.size());
        for (int i = 0; i < N; i++) {
            X.row(i) = shuffledX.row(indices[i]);
            shuffledY(i) = y(indices[i]);
        }

        // transform y to one-hot encoding
        Y = Eigen::MatrixXd::Zero(N, groups);
        for (int i = 0; i < N; ++i) Y(i, static_cast<int>(shuffledY(i))) = 1.0;
    }
}   // namespace nn