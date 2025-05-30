#ifndef MATERIALS_HPP
#define MATERIALS_HPP

#include <iostream>
#include <opencv2/opencv.hpp>

namespace materials {

    std::pair<int, int> getScreenResolution();
    std::pair<int, int> getCenterPosition(int windowWidth, int windowHeight);
    cv::Mat convertToGrayscale(const cv::Mat &input);
    void showPairImage(int widthSingle, int heightSingle,
                       const std::pair<cv::Mat, cv::Mat> &sources);
    void showImageCenterWindow(const cv::Mat &image);

}   // namespace materials

#endif   // MATERIALS_HPP