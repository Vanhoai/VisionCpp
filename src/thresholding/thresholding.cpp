#include "thresholding.hpp"
#include "../core/macros/macros.hpp"
#include <opencv2/opencv.hpp>

cv::Mat Thresholding::applyThresholding(const int thresh, const int maxVal,
                                        const ThresholdingType type) const {

    cv::Mat processedImage = getProcessed();
    if (processedImage.empty() || processedImage.channels() != 1) {
        std::cerr << "Thresholding: grayImage is not a gray image" << std::endl;
        return {};
    }

    const int width = processedImage.cols;
    const int height = processedImage.rows;
    cv::Mat result = cv::Mat::zeros(height, width, CV_8UC1);

    FOR(i, 0, processedImage.rows - 1) {
        FOR(j, 0, processedImage.cols - 1) {
            if (type == ThresholdingType::THRESH_BINARY) {
                if (processedImage.at<uchar>(i, j) >= thresh)
                    result.at<uchar>(i, j) = maxVal;
                else
                    result.at<uchar>(i, j) = 0;
            } else if (type == ThresholdingType::THRESH_BINARY_INV) {
                if (processedImage.at<uchar>(i, j) < thresh)
                    result.at<uchar>(i, j) = maxVal;
                else
                    result.at<uchar>(i, j) = 0;
            } else if (type == ThresholdingType::THRESH_TRUNC) {
                if (processedImage.at<uchar>(i, j) > thresh)
                    result.at<uchar>(i, j) = thresh;
                else
                    result.at<uchar>(i, j) = processedImage.at<uchar>(i, j);
            } else if (type == ThresholdingType::THRESH_TOZERO) {
                if (processedImage.at<uchar>(i, j) > thresh)
                    result.at<uchar>(i, j) = processedImage.at<uchar>(i, j);
                else
                    result.at<uchar>(i, j) = 0;
            } else if (type == ThresholdingType::THRESH_TOZERO_INV) {
                if (processedImage.at<uchar>(i, j) <= thresh)
                    result.at<uchar>(i, j) = processedImage.at<uchar>(i, j);
                else
                    result.at<uchar>(i, j) = 0;
            } else {
                std::cerr << "Thresholding: Unknown thresholding type"
                          << std::endl;
                return {};
            }
        }
    }

    return result;
}

int Thresholding::otsuThresholding() const {
    constexpr int N = 256;   // Number of gray levels
    cv::Mat processedImage = getProcessed();

    // calculate histogram

    int histogram[N] = {0};
    FOR(y, 0, processedImage.rows - 1) FOR(x, 0, processedImage.cols - 1) {
        const int pixelValue = processedImage.at<uchar>(y, x);
        histogram[pixelValue]++;
    }

    float P[N] = {0};
    float uT = 0;   // μT = Σ(i * P(i)) i = 0 ... 255

    const int pixels = processedImage.rows * processedImage.cols;
    FOR(i, 0, N - 1) {
        P[i] = static_cast<float>(histogram[i]) / static_cast<float>(pixels);
        uT += P[i] * static_cast<float>(i);
    }

    float maxSigma = 0;
    int bestThreshold = 0;
    float w0 = 0, sum0 = 0;

    FOR(t, 0, N - 1) {
        w0 += P[t];
        sum0 += static_cast<float>(t) * P[t];

        const float w1 = 1.0f - w0;
        if (w0 == 0 || w1 == 0)
            continue;

        const float mu0 = sum0 / w0;
        const float mu1 = (uT - sum0) / w1;

        float sigmaB = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);
        if (sigmaB > maxSigma) {
            maxSigma = sigmaB;
            bestThreshold = t;
        }
    }

    return bestThreshold;
}

int Thresholding::openCVOtsu(const int thresh, const int maxVal) const {
    const cv::Mat processedImage = getProcessed();

    cv::Mat binaryImage;
    const double threshold =
        cv::threshold(processedImage, binaryImage, thresh, maxVal,
                      cv::THRESH_BINARY + cv::THRESH_OTSU);

    return static_cast<int>(threshold);
}

void Thresholding::compareWithOpencv(const int thresh, const int maxVal) const {
    std::cout << "=== OTSU THRESHOLD COMPARISON ===" << std::endl;
    auto start = cv::getTickCount();
    const int customThreshold = otsuThresholding();

    auto customTime =
        (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;

    start = cv::getTickCount();
    int opencvThreshold = openCVOtsu(thresh, maxVal);
    auto opencvTime =
        (cv::getTickCount() - start) / cv::getTickFrequency() * 1000;

    std::cout << "Custom Implementation: " << customThreshold
              << " (Time: " << customTime << "ms)" << std::endl;

    std::cout << "OpenCV Built-in: " << opencvThreshold
              << " (Time: " << opencvTime << "ms)" << std::endl;
}
