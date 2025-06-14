//
// Created by VanHoai on 10/6/25.
//

#include "core/multivariate_normal.hpp"

#include <Eigen/Cholesky>
#include <algorithm>

namespace core {

    Eigen::MatrixXd MultivariateNormal::random(const int N, const Eigen::VectorXd &mean,
                                               const Eigen::MatrixXd &covariance) {
        const int d = mean.size();

        // Cholesky decomposition of covariance matrix
        const Eigen::LLT<Eigen::MatrixXd> choleskySolver(covariance);
        const Eigen::MatrixXd L = choleskySolver.matrixL();

        // Generate standard normal
        Eigen::MatrixXd SN(N, d);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < d; ++j) SN(i, j) = normal_distribution(generator);
        }

        // Transform to desired distribution: X = mean + L * Z
        Eigen::MatrixXd X(N, d);

        for (int i = 0; i < N; ++i) {
            Eigen::VectorXd z = SN.row(i).transpose();
            Eigen::VectorXd x = mean + L * z;
            X.row(i) = x.transpose();
        }

        return X;
    }

    std::vector<int> MultivariateNormal::shuffleIndices(const int size) {
        std::vector<int> indices(size);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), generator);
        return indices;
    }

}   // namespace core