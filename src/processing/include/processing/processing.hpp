//
// File        : processing.hpp
// Author      : Hinsun
// Date        : 2025-06-25
// Copyright   : (c) 2025 Tran Van Hoai
// License     : MIT
//

#ifndef PROCESSING_HPP
#define PROCESSING_HPP

#include <vector>

namespace processing {

class Keypoint {
public:
    float x, y;                     // Position in image
    float scale;                    // Scale (sigma)
    float angle;                    // Dominant orientation
    int octave;                     // Octave index
    int layer;                      // Layer within octave
    std::vector<float> descriptor;  // 128-dimensional descriptor
    float response;                 // Response value

    Keypoint() : x(0), y(0), scale(0), angle(0), octave(0), layer(0), response(1.0f) {
        descriptor.resize(128, 0.0f);
    }
};

}  // namespace processing

#endif  // PROCESSING_HPP
