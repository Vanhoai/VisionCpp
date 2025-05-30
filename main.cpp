#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

// Filters
#include "src/filters/edge_detection/edge_detection.hpp"
#include "src/filters/sharpen/sharpen.hpp"

// Materials
#include "src/materials.hpp"

// Thresholding
#include "src/thresholding/thresholding.hpp"

using namespace std;
using namespace cv;

const string root = "/Users/aurorastudyvn/Workspace/ML/VisionCpp";
const string image_path = root + "/image.jpg";
const string video_path = root + "/video.mp4";

int main() {
    cout << "OpenCV Version: " << CV_VERSION << endl;
    cout << "C++ Standard Version: " << __cplusplus << endl;
    // VideoCapture video_capture(video_path);
    // if (!video_capture.isOpened()) {
    //     cout << "Can not open video file: " << video_path << endl;
    //     return -1;
    // }
    //
    // const double fps = video_capture.get(CAP_PROP_FPS);
    // cout << "Video FPS: " << fps << endl;
    //
    // const double frame_count = video_capture.get(CAP_PROP_FRAME_COUNT);
    // cout << "Frame count: " << frame_count << endl;
    //
    // while (video_capture.isOpened()) {
    //     // Initialise frame matrix
    //     Mat frame;
    //     // Initialize a boolean to check if frames are there or not
    //     // If frames are present, show it
    //     if (const bool isSuccess = video_capture.read(frame);
    //         isSuccess == true) {
    //         // display frames
    //         imshow("Frame", frame);
    //     } else {
    //         cout << "Video camera is disconnected" << endl;
    //         break;
    //     }
    //
    //     // wait 20 ms between successive frames and break the loop if key q
    //     if (const int key = waitKey(20); key == 'q') {
    //         cout << "q key is pressed by the user. Stopping the video" <<
    //         endl; break;
    //     }
    // }
    //
    // video_capture.release();
    // destroyAllWindows();

    VideoCapture capture(1);
    if (!capture.isOpened()) {
        cerr << "Error: Could not open video capture." << endl;
        return -1;
    }

    const int frameWidth = static_cast<int>(capture.get(CAP_PROP_FRAME_WIDTH));
    const int frameHeight =
        static_cast<int>(capture.get(CAP_PROP_FRAME_HEIGHT));
    const Size frameSize(frameWidth, frameHeight);
    constexpr int fps = 60;

    VideoWriter output(root + "/output.avi",
                       VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frameSize);

    Mat frame;
    while (true) {
        bool isSuccess = capture.read(frame);
        if (isSuccess == false) {
            cout << "Stream disconnected" << endl;
            break;
        } else {
            output.write(frame);
            // display frames
            imshow("Frame", frame);

            if (const int key = waitKey(20); key == 27) {
                cout << "Stopping the video " << endl;
                break;
            }
        }

        if (waitKey(27) >= 0)
            break;
    }

    capture.release();
    destroyAllWindows();
    return 0;
}
