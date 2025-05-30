#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <string>

// Filters
#include "src/filters/edge_detection/edge_detection.hpp"
#include "src/filters/sharpen/sharpen.hpp"

// Materials
#include "src/materials.hpp"

// Thresholding
#include "src/thresholding/thresholding.hpp"

using namespace std;
using namespace cv;

int main() {
    cout << "OpenCV version: " << CV_VERSION << endl;
    cout << "C++ standard version: " << __cplusplus << endl;
    const string path = "/Users/aurorastudyvn/Workspace/ML/VisionCpp/image.jpg";

    // Read image with OpenCV
    const Mat imageInput = imread(path);
    Mat image;
    resize(imageInput, image, cv::Size(600, 600));

    const Thresholding thresholding(image);
    thresholding.compareWithOpencv(128, 255);

    return 0;
}
