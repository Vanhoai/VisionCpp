#include <Eigen/Core>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

// Neural Networks
#include "src/modules/nn/layer/layer.hpp"
#include "src/modules/nn/loss/loss.hpp"
#include "src/modules/nn/model/model.hpp"
#include "src/modules/nn/optimizer/optimizer.hpp"
#include "src/modules/nn/prepare/prepare.hpp"

// Core
#include "src/core/utilities/multivariate_normal/multivariate_normal.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace nn;

const string root = "/Users/aurorastudyvn/Workspace/ML/VisionCpp";
const string image_path = root + "/image.jpg";
const string video_path = root + "/video.mp4";

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
    cout << "Training with " << N << " samples, "
         << d << " features, " << classes << " classes." << endl;

    sequential.fit(X, Y, epochs, batchSize);
    return EXIT_SUCCESS;
}
