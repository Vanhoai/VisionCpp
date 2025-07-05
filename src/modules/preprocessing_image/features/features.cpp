//
// Created by VanHoai on 1/6/25.
//

#include "features.hpp"

#include <opencv2/opencv.hpp>

namespace feature_detection {

void FeatureDetector::harrisCornerDetection(const Mat &image) {}

void FeatureDetector::siftDetection(const Mat &image) {
    const Ptr<Feature2D> sift = SIFT::create();

    vector<KeyPoint> keypoints;
    Mat descriptors;
    sift->detectAndCompute(image, Mat(), keypoints, descriptors);

    Mat output;
    drawKeypoints(image, keypoints, output, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

    imshow("SIFT", output);
    waitKey(0);
    destroyAllWindows();
}

}  // namespace feature_detection