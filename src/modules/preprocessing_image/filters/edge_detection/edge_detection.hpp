#ifndef EDGE_DETECTION_HPP
#define EDGE_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <utility>
using namespace cv;

class EdgeDetection {
private:
    Mat image;
    const int sobelKernelSize = 3;  // Size of the Sobel kernel
    int kernelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };  // Sobel kernel for X direction
    int kernelY[3][3] = {
        {-1, -2, -1},
        {0,  0,  0 },
        {1,  2,  1 }
    };  // Sobel kernel for Y direction

    void calculateGradient(const Mat &grayImage, Mat &magnitude, Mat &direction);

    Mat nonMaximumSuppression(const Mat &magnitude, const Mat &direction);
    Mat hysteresisThresholding(const Mat &suppressedImage, double lowThreshold,
                               double highThreshold);

public:
    explicit EdgeDetection(Mat inputImage) : image(std::move(inputImage)) {}

    /**
     * Applies the Sobel edge detection algorithm to the input image.
     * The input image must be a single channel (grayscale) image.
     *
     * KernelX:
     *      -1 0 1
     *      -2 0 2
     *      -1 0 1
     *
     * KernelY:
     *     -1 -2 -1
     *     0  0  0
     *     1  2  1
     * @param magnitude Output image to store the magnitude of the gradient.
     * @param direction Output image to store the direction of the gradient.
     * @param dx If true, compute the gradient in the X direction.
     * @param dy If true, compute the gradient in the Y direction.
     *
     * Calculates:
     *      - Magnitude: sqrt(Gx^2 + Gy^2)
     *      - Direction: atan2(Gy, Gx)
     *
     * @return Mat The resulting image after applying Sobel edge detection.
     */
    void applySobel(Mat &magnitude, Mat &direction, bool dx = true, bool dy = true);

    /**
     * Performs Canny edge detection on the input image.
     * Follows the Canny edge detection algorithm:
     * 1. Noise Reduction
     * - Gaussian Blur is applied to reduce noise and detail in the image.
     * - Converts to grayscale to work with single-channel image
     *
     * 2. Calculating Intensity Gradient
     * - Applies Sobel operators (3×3 kernels) to detect edges in X and Y
     * directions
     * - Calculates gradient magnitude (follow the formula of sobel:
     * sqrt(Gx^2 + Gy^2))
     * - Calculates gradient direction (follow the formula of sobel:
     * atan2(Gy, Gx)) and normalizes it to 0-180 degrees
     *
     * 3. Non-Maximum Suppression (Suppression of False Edges)
     * - For each pixel, checks if it's a local maximum along the gradient
     * direction
     * - Compares with two neighbors in the gradient direction
     * - Keeps only pixels that are local maxima, suppressing others
     * - Uses 4 directional cases: horizontal (0°), diagonal (45°), vertical
     * (90°), diagonal (135°)
     *
     *
     * 4. Hysteresis Thresholding
     * - Uses two thresholds: high (strong edges) and low (weak edges)
     * - First pass: Marks pixels above high threshold as strong edges
     * (255), pixels between thresholds as weak edges (128)
     * - Second pass: Uses DFS to connect weak edges that are adjacent to
     * strong edges
     * - Third pass: Removes remaining isolated weak edges
     */
    Mat cannyEdgeDetection(double lowThreshold, double highThreshold);
};

#endif  // EDGE_DETECTION_HPP