#include "sharpen.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace sharpen {
    void test_sharpen() {
        cout << "OpenCV version: " << CV_VERSION << endl;
        cout << "C++ standard version: " << __cplusplus << endl;

        string path =
            "/Users/aurorastudyvn/Workspace/ML/CppDeepLearning/image.jpg";

        // Read image with OpenCV
        Mat imageInput = imread(path);
        Mat image;
        resize(imageInput, image, Size(600, 600));

        if (image.empty()) {
            cout << "Could not open of find the image!" << endl;
            return;
        }

        // Apply sharpening using kernel
        // Mat sharp_img;
        // Mat kernel3 = (Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
        // filter2D(image, sharp_img, -1, kernel3, Point(-1, -1), 0,
        // BORDER_DEFAULT); imshow("Original", image); imshow("Sharpened",
        // sharp_img); imwrite("sharp_image.jpg", sharp_img); waitKey();
        // destroyAllWindows();

        // Apply bilateral filtering
        Mat bilateral_filter;
        bilateralFilter(image, bilateral_filter, 9, 75, 75);
        imshow("Original", image);
        imshow("Bilateral filtering", bilateral_filter);
        imwrite("bilateral_filtering.jpg", bilateral_filter);
        waitKey(0);
        destroyAllWindows();
    }
}   // namespace sharpen