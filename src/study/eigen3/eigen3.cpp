//
// Created by VanHoai on 8/6/25.
//

#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

#include "eigen3.hpp"
// Include the main Eigen header
#include <Eigen/Core>
#include <Eigen/Dense>

// For sparse matrices (if needed)
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;

void init_matrix_vector_operation() {
    const MatrixXd A(3, 3);
    const MatrixXd B = MatrixXd::Zero(3, 3);
    const MatrixXd C = MatrixXd::Ones(3, 3);
    const MatrixXd I = MatrixXd::Identity(3, 3);
    const MatrixXd R = MatrixXd::Random(3, 3);
    MatrixXd D(2, 2);
    D << 1, 2, 3, 4;

    vector<double> vec = {1, 2, 3, 4};
    VectorXd V = Eigen::Map<VectorXd>(vec.data(), vec.size());

    cout << "A:\n" << A << endl;
    cout << "B:\n" << B << endl;
    cout << "C:\n" << C << endl;
    cout << "D:\n" << D << endl;
    cout << "I:\n" << I << endl;
    cout << "R:\n" << R << endl;
    cout << "Vector v:\n" << V << endl;
}

void operator_matrix_vector_operation() {
    // Basic Matrix Operations
    MatrixXd A(2, 3), B(3, 2), C(2, 2);
    A << 1, 2, 3, 4, 5, 6;
    B << 6, 5, 4, 3, 2, 1;

    cout << "Matrix A:\n" << A << endl;
    cout << "Matrix B:\n" << B << endl;

    // Matrix multiplication
    C = A * B;
    cout << "Matrix C (A * B):\n" << C << endl;

    // Element-wise operations (use .array())
    const MatrixXd result =
        A.array() * A.array();                  // Element-wise multiplication
    const MatrixXd sqrt_A = A.array().sqrt();   // Element-wise square root
    const MatrixXd exp_A = A.array().exp();     // Element-wise exponential

    cout << "Element-wise multiplication of A:\n" << result << endl;
    cout << "Element-wise square root of A:\n" << sqrt_A << endl;
    cout << "Element-wise exponential of A:\n" << exp_A << endl;

    // Transpose
    const MatrixXd A_T = A.transpose();
    cout << "Transpose of A:\n" << A_T << endl;

    // Addition and subtraction
    const MatrixXd sum = A + A;
    cout << "Sum of A and A:\n" << sum << endl;

    const MatrixXd diff = A - A;
    cout << "Difference of A and A:\n" << diff << endl;

    // Scalar operations
    const MatrixXd scaled = 2.0 * A;
    cout << "Scaled A (2.0 * A):\n" << scaled << endl;
}

// Activation functions
MatrixXd sigmoid(MatrixXd &Z) {
    MatrixXd A = 1.0 / (1.0 + (-Z.array()).exp());
    return A;
}

MatrixXd sigmoidDerivative(MatrixXd &Z) {
    MatrixXd dZ = sigmoid(Z);
    return dZ.array() * (1.0 - dZ.array());
}

MatrixXd relu(const MatrixXd &X) { return X.array().max(0.0); }

MatrixXd relu_derivative(const MatrixXd &x) {
    return (x.array() > 0.0).cast<double>();
}

MatrixXd softmax(MatrixXd &Z) {
    const MatrixXd exp = (Z.array() - Z.maxCoeff()).exp();
    return exp / exp.sum();
}

MatrixXd softmaxRowWise(MatrixXd &Z) {
    MatrixXd result = Z;
    VectorXd rowMax = Z.rowwise().maxCoeff();
    for (int i = 0; i < Z.rows(); i++) {
        RowVectorXd row = Z.row(i);
        RowVectorXd exps = (row.array() - rowMax(i)).exp();
        result.row(i) = exps / exps.sum();
    }

    return result;
}

// Loss functions
double mse_loss(const MatrixXd &Y, const MatrixXd &A) {
    MatrixXd diff = Y - A;
    return diff.array().square().mean();
}

double cross_entropy_loss(const MatrixXd &Y, const MatrixXd &A) {
    constexpr double epsilon = 1e-15;

    MatrixXd clipped = A.array().max(epsilon).min(1 - epsilon);
    return -(Y.array() * clipped.array().log()).sum() / Y.rows();
}

void Eigen3Learning::matrix_vector_operation() {
    init_matrix_vector_operation();
    operator_matrix_vector_operation();
}

void Eigen3Learning::essential_neural_network_operations() {
    MatrixXd A(3, 3);
    A << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    MatrixXd B(3, 3);
    B << 9, 8, 7, 6, 5, 4, 3, 2, 1;

    cout << "Matrix A:\n" << A << endl;
    cout << "Sigmoid of A:\n" << sigmoid(A) << endl;
    cout << "Sigmoid derivative of A:\n" << sigmoidDerivative(A) << endl;

    cout << "ReLU of A:\n" << relu(A) << endl;
    cout << "ReLU derivative of A:\n" << relu_derivative(A) << endl;

    cout << "Softmax of A:\n" << softmax(A) << endl;
    cout << "Softmax row-wise of A:\n" << softmaxRowWise(A) << endl;

    cout << "Cross-entropy loss between A and B: " << cross_entropy_loss(A, B)
         << endl;
    cout << "MSE loss between A and B: " << mse_loss(A, B) << endl;
}

void Eigen3Learning::matrix_manipulation() {
    // Resize matrix
    MatrixXd A(2, 3);
    A.resize(3, 2);

    cout << "Matrix A after resizing:\n" << A << endl;
    A << 1, 2, 3, 4, 5, 6;   // Fill with values

    // Row and column operations
    VectorXd row = A.row(0);               // Get first row
    VectorXd col = A.col(0);               // Get first column
    A.row(0) = VectorXd::Zero(A.cols());   // Set row to zero

    cout << "Matrix A after manipulation:\n" << A << endl;

    // Sum operations
    double total_sum = A.sum();
    VectorXd col_sums = A.colwise().sum();
    VectorXd row_sums = A.rowwise().sum();

    cout << "Row sums of A:\n" << row_sums.transpose() << endl;
    cout << "Column sums of A:\n" << col_sums.transpose() << endl;

    // Mean operations
    double mean_val = A.mean();
    VectorXd col_means = A.colwise().mean();
    VectorXd row_means = A.rowwise().mean();

    cout << "Mean of A: " << mean_val << endl;
    cout << "Row means of A:\n" << row_means.transpose() << endl;

    // Min/Max operations
    double min_val = A.minCoeff();
    double max_val = A.maxCoeff();
    VectorXd col_max = A.colwise().maxCoeff();
    VectorXd row_min = A.rowwise().minCoeff();

    cout << "Min value of A: " << min_val << endl;
    cout << "Max value of A: " << max_val << endl;

    // Norm operations
    double frobenius_norm = A.norm();
    VectorXd col_norms = A.colwise().norm();

    cout << "Frobenius norm of A: " << frobenius_norm << endl;
    cout << "Column norms of A:\n" << col_norms.transpose() << endl;
}

class MultivariateNormal {
    private:
        std::mt19937 generator;
        std::normal_distribution<double> normal_dist;

    public:
        MultivariateNormal() : normal_dist(0.0, 1.0) {
            // Set seed for reproducibility (equivalent to np.random.seed(5))
            generator.seed(5);
        }
        // Generate multivariate normal samples
        MatrixXd sample(const VectorXd &mean, const MatrixXd &cov,
                        int num_samples) {
            int dim = mean.size();
            // Cholesky decomposition of covariance matrix
            LLT<MatrixXd> chol_solver(cov);
            MatrixXd L = chol_solver.matrixL();
            // Generate standard normal samples
            MatrixXd samples(num_samples, dim);
            for (int i = 0; i < num_samples; ++i) {
                for (int j = 0; j < dim; ++j) {
                    samples(i, j) = normal_dist(generator);
                }
            }
            // Transform to desired distribution: X = mean + L * Z
            MatrixXd result(num_samples, dim);
            for (int i = 0; i < num_samples; ++i) {
                VectorXd z = samples.row(i).transpose();
                VectorXd x = mean + L * z;
                result.row(i) = x.transpose();
            }

            return result;
        }

        // Shuffle indices randomly
        std::vector<int> shuffle_indices(int size) {
            std::vector<int> indices(size);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), generator);
            return indices;
        }
};

void Eigen3Learning::random_matrix_operations() {

    int num = 50;     // Number of samples per group
    int groups = 4;   // Number of groups

    std::vector<VectorXd> means(4);
    means[0] = VectorXd(2);
    means[0] << 1, 1;
    means[1] = VectorXd(2);
    means[1] << 1, 6;
    means[2] = VectorXd(2);
    means[2] << 6, 1;
    means[3] = VectorXd(2);
    means[3] << 6, 6;

    MatrixXd cov(2, 2);
    cov << 1, 0, 0, 1;

    // Initialize data matrices
    MatrixXd X(num * groups, 2);
    VectorXd y(num * groups);

    // Create multivariate normal generator
    MultivariateNormal mvn;
    // Generate data for each group
    for (int idx = 0; idx < groups; ++idx) {
        MatrixXd points = mvn.sample(means[idx], cov, num);
        // Fill X matrix
        X.block(idx * num, 0, num, 2) = points;
        // Fill y vector
        for (int i = 0; i < num; ++i) {
            y(idx * num + i) = idx;
        }
    }
    // Transpose X to match Python format (features x samples)
    MatrixXd datas = X.transpose();
    // Shuffle the data
    std::vector<int> indices = mvn.shuffle_indices(datas.cols());
    // Apply shuffling
    MatrixXd shuffled_datas(datas.rows(), datas.cols());
    VectorXd shuffled_labels(y.size());
    for (int i = 0; i < indices.size(); ++i) {
        shuffled_datas.col(i) = datas.col(indices[i]);
        shuffled_labels(i) = y(indices[i]);
    }

    cout << "Shuffled Data:\n" << shuffled_datas << endl;
    cout << "Shuffled Labels:\n" << shuffled_labels.transpose() << endl;
}