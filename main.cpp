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

using namespace std;
using namespace cv;
using namespace Eigen;

const string root = "/Users/aurorastudyvn/Workspace/ML/VisionCpp";
const string image_path = root + "/image.jpg";
const string video_path = root + "/video.mp4";

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);

    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);

    Eigen3Learning::random_matrix_operations();
    return EXIT_SUCCESS;
}
