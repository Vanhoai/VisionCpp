#include "haar_cascade.hpp"

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace detectors {
void HaarCascadeDetector::realtime_with_opencv() const {
    CascadeClassifier faceDetector(face_cascade_path);
    const string windowName = "Face Detection";
    namedWindow(windowName);

    VideoCapture videoCapture(1);
    if (!videoCapture.isOpened()) {
        cout << "Could not open video capture" << endl;
        return;
    }

    Mat frame;
    vector<Rect> faces;
    while (true) {
        if (const bool isSuccess = videoCapture.read(frame); !isSuccess) break;

        faceDetector.detectMultiScale(frame, faces, 1.1, 4, CASCADE_SCALE_IMAGE, Size(30, 30));

        for (const auto &face : faces) {
            const Point center(face.x + face.width * 0.5, face.y + face.height * 0.5);

            ellipse(frame, center, Size(face.width * 0.5, face.height * 0.5), 0, 0, 360,
                    Scalar(0, 255, 0), 4, 8, 0);

            const int horizontal = (face.x + face.width * 0.5);
            const int vertical = (face.y + face.width * 0.5);
            cout << "Position of the face is:" << "(" << horizontal << "," << vertical << ")"
                 << endl;
        }

        imshow(windowName, frame);
        if (waitKey(30) >= 0) break;
    }

    videoCapture.release();
    destroyAllWindows();
}

}  // namespace detectors