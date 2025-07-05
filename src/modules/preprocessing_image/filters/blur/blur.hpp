#ifndef BLUR_HPP
#define BLUR_HPP

#include <opencv2/opencv.hpp>

namespace blur {
void applyKernel(cv::Mat &inputImage, cv::Mat &outputImage, const cv::Mat &kernel);
}

#endif
