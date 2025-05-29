#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#include "src/filters/edge_detection/edge_detection.hpp"
#include "src/filters/sharpen/sharpen.hpp"
#include "src/materials.hpp"

using namespace std;
using namespace cv;

int main() {
    cout << "OpenCV version: " << CV_VERSION << endl;
    cout << "C++ standard version: " << __cplusplus << endl;

    string path = "/Users/aurorastudyvn/Workspace/ML/CppDeepLearning/image.jpg";

    // Read image with OpenCV
    Mat imageInput = imread(path);
    Mat image;
    resize(imageInput, image, cv::Size(600, 600));

    if (image.empty()) {
        cout << "Could not open of find the image!" << endl;
        return -1;
    }

    // Mat sobel_x, sobel_y, sobel_xy;
    // Sobel(gray, sobel_x, CV_64F, 1, 0, 5);
    // Sobel(gray, sobel_y, CV_64F, 0, 1, 5);
    // Sobel(gray, sobel_xy, CV_64F, 1, 1, 5);

    // // Display Sobel edge detection images
    // imshow("Sobel X", sobel_x);
    // waitKey(0);
    // destroyAllWindows();

    // imshow("Sobel Y", sobel_y);
    // waitKey(0);
    // destroyAllWindows();

    // imshow("Sobel XY using Sobel() function", sobel_xy);
    // waitKey(0);
    // destroyAllWindows();
    // materials::showPairImage(600, 600, make_pair(image, grayImage));

    EdgeDetection edgeDetection(image);
    Mat detected = edgeDetection.cannyEdgeDetection(50, 150);

    imshow("Canny Edge Detection", detected);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
