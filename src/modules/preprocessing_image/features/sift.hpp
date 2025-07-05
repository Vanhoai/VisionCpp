//
// Created by VanHoai on 1/6/25.
//

#ifndef SIFT_HPP
#define SIFT_HPP

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * SIFT (Scale-Invariant Feature Transform): this is a algorithms to detect and
 * describe local features in images. Depending on 3 things:
 * 1. Scale
 * 2. Rotation
 * 3. Illumination
 *
 * Following steps:
 * 1. Scale-space Extrema Detection
 * 2. Keypoint Localization
 * 3. Orientation Assignment
 * 4. Keypoint Descriptor
 */

using namespace cv;
using namespace std;

namespace feature_detection {

class SIFTDetector {
private:
    int octaves;
    int scales;
    double sigma;
    double contrastThreshold;
    double edgeThreshold;

    struct Keypoint {
        float x, y;
        float scale;
        float angle;
        int octave;
        std::vector<float> descriptor;
    };

public:
    explicit SIFTDetector(const int octaves = 4, const int scales = 3, const double sigma = 1.6,
                          const double contrastThreshold = 0.04, const double edgeThreshold = 10.0)
        : octaves(octaves),
          scales(scales),
          sigma(sigma),
          contrastThreshold(contrastThreshold),
          edgeThreshold(edgeThreshold) {}

    void detect(const Mat &image, std::vector<Keypoint> &keypoints);

    static Mat createGaussianKernel(double sigma, int size = 0);

    [[nodiscard]] vector<vector<Mat>> buildScaleSpace(const Mat &image) const;

    [[nodiscard]] vector<vector<Mat>> buildDoG(const vector<vector<Mat>> &scaleSpace) const;

    static bool isExtremum(const vector<vector<Mat>> &dogSpace, int octave, int scale, int x,
                           int y);

    bool isValidKeypoint(const vector<vector<Mat>> &dogSpace, int octave, int scale, int x, int y);
};

}  // namespace feature_detection

#endif  // SIFT_HPP
