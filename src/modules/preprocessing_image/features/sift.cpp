//
// Created by VanHoai on 1/6/25.
//

#include "sift.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

namespace feature_detection {

    Mat SIFTDetector::createGaussianKernel(double sigma, int size) {
        if (size == 0)
            size = 2 * ceil(3 * sigma) + 1;

        Mat kernel(size, size, CV_32F);
        double sum = 0.0;
        const int center = size / 2;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                const double x = i - center;
                const double y = j - center;
                const double value = exp(-(x * x + y * y) / (2 * sigma * sigma));

                kernel.at<float>(i, j) = static_cast<float>(value);
                sum += value;
            }
        }

        kernel /= sum;
        return kernel;
    }

    vector<vector<Mat>> SIFTDetector::buildScaleSpace(const Mat &image) const {
        vector<vector<Mat>> scaleSpace(octaves);
        Mat currentImage = image.clone();

        for (int o = 0; o < octaves; o++) {
            scaleSpace[o].resize(scales + 3);
            scaleSpace[o][0] = currentImage.clone();   // original image

            for (int s = 1; s < scales + 3; s++) {
                const double currentSigma = sigma * pow(2.0, s / static_cast<double>(scales));
                Mat kernel = createGaussianKernel(currentSigma);
                filter2D(scaleSpace[o][s - 1], scaleSpace[o][s], -1, kernel);
            }

            if (o < octaves - 1)
                resize(scaleSpace[o][scales], currentImage,
                       Size(currentImage.cols / 2, currentImage.rows / 2));
        }

        return scaleSpace;
    }

    vector<vector<Mat>> SIFTDetector::buildDoG(const vector<vector<Mat>> &scaleSpace) const {
        vector<vector<Mat>> dogSpace(octaves);

        for (int o = 0; o < octaves; o++) {
            dogSpace[o].resize(scales + 2);
            for (int s = 0; s < scales + 2; s++) {
                dogSpace[o][s] = scaleSpace[o][s + 1] - scaleSpace[o][s];
            }
        }

        return dogSpace;
    }

    bool SIFTDetector::isExtremum(const vector<vector<Mat>> &dogSpace, const int octave,
                                  const int scale, const int x, const int y) {
        const float centerValue = dogSpace[octave][scale].at<float>(y, x);
        bool isMax = true, isMin = true;
        for (int ds = -1; ds <= 1; ds++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (ds == 0 && dy == 0 && dx == 0)
                        continue;

                    const float neighborValue =
                        dogSpace[octave][scale + ds].at<float>(y + dy, x + dx);
                    if (centerValue <= neighborValue)
                        isMax = false;
                    if (centerValue >= neighborValue)
                        isMin = false;

                    if (!isMax && !isMin)
                        return false;
                }
            }
        }

        return isMax || isMin;
    }

    bool SIFTDetector::isValidKeypoint(const vector<vector<Mat>> &dogSpace, int octave, int scale,
                                       int x, int y) {
        float value = dogSpace[octave][scale].at<float>(y, x);
        if (abs(value) < contrastThreshold)
            return false;

        float dxx = dogSpace[octave][scale].at<float>(y, x + 1) +
                    dogSpace[octave][scale].at<float>(y, x - 1) - 2 * value;
        float dyy = dogSpace[octave][scale].at<float>(y + 1, x) +
                    dogSpace[octave][scale].at<float>(y - 1, x) - 2 * value;
        float dxy = (dogSpace[octave][scale].at<float>(y + 1, x + 1) -
                     dogSpace[octave][scale].at<float>(y + 1, x - 1) -
                     dogSpace[octave][scale].at<float>(y - 1, x + 1) +
                     dogSpace[octave][scale].at<float>(y - 1, x - 1)) /
                    4.0;

        float trace = dxx + dyy;
        float det = dxx * dyy - dxy * dxy;
        if (det <= 0)
            return false;
        float ratio = trace * trace / det;
        float threshold = (edgeThreshold + 1) * (edgeThreshold + 1) / edgeThreshold;

        return ratio < threshold;
    }

}   // namespace feature_detection
