#include <Eigen/Core>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Filters
#include "src/filters/edge_detection/edge_detection.hpp"
#include "src/filters/sharpen/sharpen.hpp"

// Materials
#include "src/materials.hpp"

// Thresholding
#include "src/thresholding/thresholding.hpp"

// Detectors
#include "src/detectors/haar_cascade/haar_cascade.hpp"
#include "src/detectors/hog/hog.hpp"

// Features
#include "src/features/features.hpp"
#include "src/features/sift.hpp"

// Neural Networks
#include "src/eigen3/eigen3.hpp"
#include "src/nn/activation/activation.hpp"
#include "src/nn/layer/layer.hpp"
#include "src/nn/loss/loss.hpp"
#include "src/nn/model/model.hpp"
#include "src/nn/optimizer/optimizer.hpp"

// Core
#include "src/core/utilities/multivariate_normal/multivariate_normal.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace nn;

const string root = "/Users/aurorastudyvn/Workspace/ML/VisionCpp";
const string image_path = root + "/image.jpg";
const string video_path = root + "/video.mp4";

void prepare(const int N, const int groups, const int d, MatrixXd &X,
             MatrixXd &Y) {
    X = MatrixXd::Zero(N, d);
    Y = MatrixXd::Zero(N, groups);

    MatrixXd means(4, 2);
    means << 1, 1, 1, 6, 6, 1, 6, 6;
    MatrixXd covariance(2, 2);
    covariance << 1, 0, 0, 1;

    const int P = N / groups;

    utilities::MultivariateNormal mvn;

    VectorXd y(N);
    MatrixXd shuffledX(X.rows(), X.cols());

    for (int i = 0; i < groups; ++i) {
        const MatrixXd Xi = mvn.random(means.row(i), covariance, P);
        shuffledX.block(i * P, 0, P, d) = Xi;
        for (int j = 0; j < P; ++j)
            y(i * P + j) = i;
    }

    const vector<int> indices = mvn.shuffleIndices(N);

    VectorXd shuffledY(y.size());
    for (int i = 0; i < N; i++) {
        X.row(i) = shuffledX.row(indices[i]);
        shuffledY(i) = y(indices[i]);
    }

    // transform y to one-hot encoding
    Y = MatrixXd::Zero(N, groups);
    for (int i = 0; i < N; ++i)
        Y(i, static_cast<int>(shuffledY(i))) = 1.0;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    constexpr int N = 1000;
    int d = 2;
    int d1 = 5;
    int classes = 4;

    MatrixXd X(N, d);
    MatrixXd Y(N, classes);
    prepare(N, classes, d, X, Y);

    vector<unique_ptr<Layer>> layers;
    layers.push_back(make_unique<ReLU>(d, d1));
    layers.push_back(make_unique<Softmax>(d1, classes));

    unique_ptr<Loss> loss = make_unique<CrossEntropyLoss>();
    unique_ptr<Optimizer> optimizer = make_unique<SGD>(1e-3);
    nn::Sequential sequential(layers, loss, optimizer);

    constexpr int epochs = 10000;
    constexpr int batchSize = 256;

    sequential.fit(X, Y, epochs, batchSize);

    return EXIT_SUCCESS;
}
