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
#include "src/nn/activation.hpp"
#include "src/nn/layer.hpp"
#include "src/nn/loss.hpp"
#include "src/nn/model.hpp"
#include "src/nn/optimizer.hpp"

// Utilities
#include "src/utilities/multivariate_normal/multivariate_normal.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace nn;

const string root = "/Users/aurorastudyvn/Workspace/ML/VisionCpp";
const string image_path = root + "/image.jpg";
const string video_path = root + "/video.mp4";

void prepare() {
    MatrixXd means(4, 2);
    means << 1, 1, 1, 6, 6, 1, 6, 6;
    MatrixXd covariance(2, 2);
    covariance << 1, 0, 0, 1;

    utilities::MultivariateNormal mvn;
    constexpr int N = 10;
    const MatrixXd X1 = mvn.random(means.row(0), covariance, N);
    const MatrixXd X2 = mvn.random(means.row(1), covariance, N);
    const MatrixXd X3 = mvn.random(means.row(2), covariance, N);
    const MatrixXd X4 = mvn.random(means.row(3), covariance, N);

    cout << "X1:\n" << X1 << endl;
    cout << "X2:\n" << X2 << endl;
    cout << "X3:\n" << X3 << endl;
    cout << "X4:\n" << X4 << endl;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    prepare();

    int d = 784;
    int d1 = 1000;
    int d2 = 512;
    int classes = 10;

    vector<unique_ptr<Layer>> layers;
    layers.push_back(make_unique<ReLU>(d, d1));
    layers.push_back(make_unique<ReLU>(d1, d2));
    layers.push_back(make_unique<Softmax>(d2, classes));

    unique_ptr<Loss> loss = make_unique<CrossEntropyLoss>();
    unique_ptr<Optimizer> optimizer = make_unique<SGD>(1e-3);
    nn::Sequential sequential(layers, loss, optimizer);

    return EXIT_SUCCESS;
}
