//
// Created by VanHoai on 10/6/25.
//

#ifndef MULTIVARIATE_NORMAL_HPP
#define MULTIVARIATE_NORMAL_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using namespace Eigen;

namespace utilities {

    class MultivariateNormal {
        private:
            mt19937 generator;
            normal_distribution<> normal_distribution;

        public:
            explicit MultivariateNormal(const double mean = 0.0,
                                        const double standardDeviation = 1.0)
                : normal_distribution(mean, standardDeviation) {
                std::random_device dev;
                generator = std::mt19937(dev());
            }

            /**
             * Generates random samples from a multivariate normal distribution.
             *
             * @param mean The mean vector of the distribution.
             * @param covariance The covariance matrix of the distribution.
             * @param N The number of samples to generate.
             * @return A matrix where each row is a sample from the
             * distribution.
             *
             * @steps
             * 1. Perform Cholesky decomposition of the covariance matrix. In
             * this step, covariance matrix is decomposed into a lower
             * triangular with formula: L * L^T = covariance.
             * 2. Generate standard normal random variables. This is done by
             * filling a matrix with random numbers drawn from a standard normal
             * distribution.
             * 3. Transform Z to the desired distribution by 𝒩(μ, Σ)
             * formula: x = mean + L * z
             */
            MatrixXd random(const VectorXd &mean, const MatrixXd &covariance,
                            const int N);

            std::vector<int> shuffleIndices(const int size);
    };

}   // namespace utilities

#endif   // MULTIVARIATE_NORMAL_HPP
