#ifndef HOG_H
#define HOG_H

/**
 * Some keywords:
 * 1. Feature Descriptor: This is a method for extracting features from images -
 * (refs: HOG, SIFT, SURF, etc.).
 *
 * 2. Histogram: This is a graph representing the distribution of pixel
 * intensities/
 *
 * 3. Gradient: This is gradient of intensity vectors in an image. It helps
 * detect directions of edges in an image.
 *
 * 4. Local Cell: This is a small region in an image used to compute.
 *
 * 5. Local Portion (Block): This is a larger region in an image used to
 * compute.
 *
 * 6. Local Normalization: This is a process of adjusting the local intensity.
 * As usual, it is divided norm 2 normalization and norm 1 normalization.
 *
 * 7. Gradient Direction: This is a value calculated from the gradient vector x
 * and y. It helps determine the direction of colors intensity in an image.
 *                    O = atan2(gy / gx)
 *
 * 8. Gradient Magnitude: This is a distance of the gradient vector by x & y
 * direction. It calculates following formula:
 *                    M = sqrt(gx^2 + gy^2)
 */
namespace detectors {

    class HogDetector {};

}   // namespace detectors

#endif
