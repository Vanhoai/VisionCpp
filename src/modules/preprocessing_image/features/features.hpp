//
// Created by VanHoai on 1/6/25.
//

#ifndef FEATURES_HPP
#define FEATURES_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <vector>

using namespace cv;
using namespace std;

namespace feature_detection {

class FeatureDetector {
public:
    static void harrisCornerDetection(const Mat &image);
    static void siftDetection(const Mat &image);
};

}  // namespace feature_detection

#endif  // FEATURES_HPP