#ifndef HAAR_CASCADE_H
#define HAAR_CASCADE_H

#include <iostream>
#include <string>

namespace detectors {

class HaarCascadeDetector {
private:
    const std::string face_cascade_path =
        "/opt/homebrew/Caskroom/miniconda/base/pkgs/"
        "libopencv-4.10.0-headless_py311h09a821a_13/share/opencv4/"
        "haarcascades/"
        "haarcascade_frontalface_default.xml";

public:
    void realtime_with_opencv() const;
};

}  // namespace detectors

#endif  // HAAR_CASCADE_H
