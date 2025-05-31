#include "blur.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace blur {
    void applyKernel(cv::Mat &inputImage, cv::Mat &outputImage,
                     const cv::Mat &kernel) {
        for (int i = 0; i < inputImage.rows; ++i) {
            for (int j = 0; j < inputImage.cols; ++j) {
                cv::Vec3b pixelValue(0, 0, 0);
                for (int ki = -kernel.rows / 2; ki <= kernel.rows / 2; ++ki) {
                    for (int kj = -kernel.cols / 2; kj <= kernel.cols / 2;
                         ++kj) {
                        int ni = i + ki;
                        int nj = j + kj;
                        if (ni >= 0 && ni < inputImage.rows && nj >= 0 &&
                            nj < inputImage.cols) {
                            pixelValue +=
                                inputImage.at<cv::Vec3b>(ni, nj) *
                                kernel.at<double>(ki + kernel.rows / 2,
                                                  kj + kernel.cols / 2);
                        }
                    }
                }
                outputImage.at<Vec3b>(i, j) = pixelValue;
            }
        }

        std::cout << "Applied kernel to image." << std::endl;
    }

    void test() {
        std::cout << "OpenCV version: " << CV_VERSION << endl;
        std::cout << "C++ standard version: " << __cplusplus << endl;

        std::string path =
            "/Users/aurorastudyvn/Workspace/ML/CppDeepLearning/image.jpg";

        // Read image with OpenCV
        Mat imageInput = imread(path);
        Mat image;
        resize(imageInput, image, Size(600, 600));

        if (image.empty()) {
            std::cout << "Could not open of find the image!" << endl;
            return;
        }

        std::cout << "Rows: " << image.rows << endl;
        std::cout << "Cols: " << image.cols << endl;

        // Mat kernel1 = (Mat_<double>(3, 3) << 0, 0, 0, 0, 1, 0, 0, 0, 0);
        // Mat identity;
        // filter2D(image, identity, -1, kernel1, Point(-1, -1), 0, 4);
        // imshow("Original", image);
        // imshow("Identity", identity);
        // imwrite("identity.jpg", identity);
        // waitKey();
        // destroyAllWindows();

        // Mat kernel2 = Mat::ones(5, 5, CV_64F);

        // // Normalize the elements
        // kernel2 = kernel2 / 25;

        // std::cout << kernel2 << endl;

        // Mat img;
        // filter2D(image, img, -1, kernel2, Point(-1, -1), 0, 4);
        // imshow("Original", image);
        // imshow("Kernel blur", img);
        // imwrite("kernel_blur.jpg", img);
        // waitKey();
        // destroyAllWindows();

        // Blurred using OpenCV C++ blur() function
        // Mat img_blur;
        // blur(image, img_blur, Size(8, 8));
        // imshow("Original", image);
        // imshow("Blurred", img_blur);
        // imwrite("blur.jpg", img_blur);
        // waitKey();
        // destroyAllWindows();

        // Performing Gaussian Blur
        // Mat gaussian_blur;
        // GaussianBlur(image, gaussian_blur, Size(5, 5), 0.5, 20);
        // imshow("Original", image);
        // imshow("Gaussian Blurred", gaussian_blur);
        // imwrite("gaussian_blur.jpg", gaussian_blur);
        // waitKey();
        // destroyAllWindows();

        // Apply Median Blur
        Mat median_blurred;
        medianBlur(image, median_blurred, 5);

        imshow("Original", image);
        imshow("Median Blurred", median_blurred);
        imwrite("median_blur.jpg", median_blurred);
        waitKey();
        destroyAllWindows();
    }
}   // namespace blur
