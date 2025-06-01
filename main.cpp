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

using namespace std;
using namespace cv;

const string root = "/Users/aurorastudyvn/Workspace/ML/VisionCpp";
const string image_path = root + "/image.jpg";
const string video_path = root + "/video.mp4";

int main() {
    cout << "OpenCV Version: " << CV_VERSION << endl;
    cout << "C++ Standard Version: " << __cplusplus << endl;

    const Mat imr = imread(image_path);
    Mat image;
    resize(imr, image, Size(600, 600));

    vector<KeyPoint> keypoints;
    const Ptr<SIFT> sift = SIFT::create();
    sift->detect(image, keypoints);

    Mat output;
    drawKeypoints(image, keypoints, output, Scalar::all(-1),
                  DrawMatchesFlags::DEFAULT);
    materials::showImageCenterWindow(output);

    return EXIT_SUCCESS;
}
