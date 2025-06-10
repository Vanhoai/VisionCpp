//
// Created by VanHoai on 10/6/25.
//

#include "multivariate_normal.hpp"

namespace utilities {

    MatrixXd MultivariateNormal::random(const VectorXd &mean,
                                        const MatrixXd &covariance,
                                        const int N) {
        const unsigned d = mean.size();

        // Cholesky decomposition of covariance matrix
        const LLT<MatrixXd> cholesky_solver(covariance);
        const MatrixXd L = cholesky_solver.matrixL();

        // Generate standard normal
        MatrixXd SN(N, d);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < d; ++j)
                SN(i, j) = normal_distribution(generator);
        }

        // Transform to desired distribution: X = mean + L * Z
        MatrixXd X(N, d);

        for (int i = 0; i < N; ++i) {
            VectorXd z = SN.row(i).transpose();
            VectorXd x = mean + L * z;
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

}   // namespace utilities