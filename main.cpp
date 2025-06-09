#include <Eigen/Core>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

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
#include "src/nn/loss.hpp"

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

    MatrixXd Y = MatrixXd::Random(3, 3);
    MatrixXd A = MatrixXd::Random(3, 3);

    CrossEntropyLoss crossEntropyLoss;
    const double lossValue = crossEntropyLoss(Y, A);
    const MatrixXd dA = crossEntropyLoss.derivative(Y, A);
    cout << "Loss value: " << lossValue << endl;
    cout << "Derivative of loss: " << dA << endl;
    return EXIT_SUCCESS;
}
