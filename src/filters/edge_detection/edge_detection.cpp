#include "edge_detection.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void EdgeDetection::calculateGradient(const Mat &grayImage, Mat &magnitude,
                                      Mat &direction) {
    int rows = grayImage.rows;
    int cols = grayImage.cols;

    // Initialize output matrices
    magnitude = Mat::zeros(rows, cols, CV_64F);
    direction = Mat::zeros(rows, cols, CV_64F);

    // Apply Sobel operators
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            double gx = 0, gy = 0;

            // Apply Sobel X kernel
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    const double pixelValue =
                        grayImage.at<uchar>(i + ki, j + kj);
                    gx += pixelValue * kernelX[ki + 1][kj + 1];
                    gy += pixelValue * kernelY[ki + 1][kj + 1];
                }
            }

            // Calculate magnitude and direction
            magnitude.at<double>(i, j) = sqrt(gx * gx + gy * gy);
            direction.at<double>(i, j) = atan2(gy, gx) * 180.0 / M_PI;

            // Normalize direction to 0-180 degrees
            if (direction.at<double>(i, j) < 0)
                direction.at<double>(i, j) += 180.0;
        }
    }
}

Mat EdgeDetection::nonMaximumSuppression(const Mat &magnitude,
                                         const Mat &direction) {
    int rows = magnitude.rows;
    int cols = magnitude.cols;

    Mat suppressed = Mat::zeros(rows, cols, CV_64F);

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            double angle = direction.at<double>(i, j);
            double mag = magnitude.at<double>(i, j);

            double neighbor1 = 0, neighbor2 = 0;

            // Determine neighbors based on gradient direction
            if ((angle >= 0 && angle < 22.5) ||
                (angle >= 157.5 && angle <= 180)) {
                // Horizontal direction (0 degrees)
                neighbor1 = magnitude.at<double>(i, j + 1);
                neighbor2 = magnitude.at<double>(i, j - 1);
            } else if (angle >= 22.5 && angle < 67.5) {
                // Diagonal direction (45 degrees)
                neighbor1 = magnitude.at<double>(i - 1, j + 1);
                neighbor2 = magnitude.at<double>(i + 1, j - 1);
            } else if (angle >= 67.5 && angle < 112.5) {
                // Vertical direction (90 degrees)
                neighbor1 = magnitude.at<double>(i - 1, j);
                neighbor2 = magnitude.at<double>(i + 1, j);
            } else if (angle >= 112.5 && angle < 157.5) {
                // Diagonal direction (135 degrees)
                neighbor1 = magnitude.at<double>(i - 1, j - 1);
                neighbor2 = magnitude.at<double>(i + 1, j + 1);
            }

            // Keep pixel if it's a local maximum
            if (mag >= neighbor1 && mag >= neighbor2)
                suppressed.at<double>(i, j) = mag;
        }
    }

    return suppressed;
}

// Helper function for hysteresis thresholding - DFS to connect weak edges
void connectWeakEdges(Mat &result, Mat &visited, int i, int j) {
    if (i < 0 || i >= result.rows || j < 0 || j >= result.cols ||
        visited.at<uchar>(i, j) == 1) {
        return;
    }

    visited.at<uchar>(i, j) = 1;

    // Check 8-connected neighbors
    for (int di = -1; di <= 1; di++) {
        for (int dj = -1; dj <= 1; dj++) {
            if (di == 0 && dj == 0)
                continue;

            int ni = i + di;
            int nj = j + dj;

            if (ni >= 0 && ni < result.rows && nj >= 0 && nj < result.cols) {
                if (result.at<uchar>(ni, nj) == 128) {   // Weak edge
                    result.at<uchar>(ni, nj) = 255;   // Convert to strong edge
                    connectWeakEdges(result, visited, ni, nj);
                }
            }
        }
    }
}

Mat EdgeDetection::hysteresisThresholding(const Mat &suppressedImage,
                                          const double lowThreshold,
                                          const double highThreshold) {
    int rows = suppressedImage.rows;
    int cols = suppressedImage.cols;

    Mat result = Mat::zeros(rows, cols, CV_8U);
    Mat visited = Mat::zeros(rows, cols, CV_8U);

    // First pass: Mark strong edges
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double pixel = suppressedImage.at<double>(i, j);
            if (pixel >= highThreshold)
                result.at<uchar>(i, j) = 255;   // Strong edge
            else if (pixel >= lowThreshold)
                result.at<uchar>(i, j) = 128;   // Weak edge
        }
    }

    // Second pass: Connect weak edges to strong edges
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols; j++) {
            if (result.at<uchar>(i, j) == 255 && !visited.at<uchar>(i, j))
                connectWeakEdges(result, visited, i, j);   // DFS
        }
    }

    // Third pass: Remove remaining weak edges
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (result.at<uchar>(i, j) == 128)
                result.at<uchar>(i, j) = 0;
        }
    }

    return result;
}

void EdgeDetection::applySobel(Mat &magnitude, Mat &direction, bool dx,
                               bool dy) {
    if (image.empty()) {
        cerr << "Input image is empty!" << endl;
        return;
    }

    if (image.channels() != 1) {
        cerr << "Input image must be a single channel (grayscale) image!"
             << endl;
        return;
    }

    // Initialize output images
    magnitude = Mat::zeros(image.size(), CV_8UC1);
    direction = Mat::zeros(image.size(), CV_32FC1);

    for (int y = 1; y < image.rows - 1; y++) {
        for (int x = 1; x < image.cols - 1; x++) {
            int gx = 0, gy = 0;

            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    int pixel = image.at<uchar>(y + j, x + i);
                    gx += kernelX[j + 1][i + 1] * pixel;
                    gy += kernelY[j + 1][i + 1] * pixel;
                }
            }

            if (!dx)
                gx = 0;   // Ignore X gradient if dx is false

            if (!dy)
                gy = 0;   // Ignore Y gradient if dy is false

            // Calculate magnitude and direction
            int mag = static_cast<int>(sqrt(gx * gx + gy * gy));
            mag = min(255, max(0, mag));
            magnitude.at<uchar>(y, x) = static_cast<uchar>(mag);

            float angle = atan2(gy, gx) * 180.0 / CV_PI;
            direction.at<float>(y, x) = angle;
        }
    }
}

Mat EdgeDetection::cannyEdgeDetection(double lowThreshold,
                                      double highThreshold) {
    if (image.empty()) {
        cerr << "Input image is empty!" << endl;
        return Mat();
    }

    // Step 1: Noise Reduction
    Mat imageBlur;
    GaussianBlur(image, imageBlur, Size(3, 3), 0, 0);
    Mat grayImage;
    cvtColor(imageBlur, grayImage, COLOR_BGR2GRAY);

    // Step 2: Calculate Intensity Gradient
    Mat gradientMagnitude, gradientDirection;
    calculateGradient(grayImage, gradientMagnitude, gradientDirection);

    // Step 3: Non-Maximum Suppression
    Mat suppressedImage =
        nonMaximumSuppression(gradientMagnitude, gradientDirection);

    // Step 4: Hysteresis Thresholding
    Mat cannyResult =
        hysteresisThresholding(suppressedImage, lowThreshold, highThreshold);

    return cannyResult;
}