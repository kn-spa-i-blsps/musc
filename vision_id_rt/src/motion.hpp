// Camera motion: image shift from phase correlation, and the homography of a pure camera rotation.
#pragma once
#include <complex>
#include <cstdint>
#include <vector>

namespace ledrx {

struct Homography {
    double h[9];
    void apply(double x, double y, double& ox, double& oy) const;
    Homography inverse() const;
    static Homography translation(double dx, double dy);
};

// Image motion caused by rotating the camera by `rot` (rotation vector, rad, camera axes x right, y down,
// z forward): maps previous-frame pixels to current-frame pixels, H = K R^T K^-1.
Homography rotation_homography(const double rot[3], int w, int h, double hfov_deg);

// dst(x, y) = src(H^-1 (x, y)), bilinear. Outside the source: nearest edge (replicate) or 0.
// Areas whose source values are all <= zero_below come out as 0.
void warp(const float* src, float* dst, int w, int h, const Homography& H, bool replicate, float zero_below = 0.0f);
void warp(const uint8_t* src, float* dst, int w, int h, const Homography& H, bool replicate, float zero_below = 0.0f);

// Global shift between consecutive frames (like cv2.phaseCorrelate on a 4x smaller grayscale image).
class PhaseCorrelator {
public:
    // Returns false for the first frame. dx, dy: how far the content moved since the previous frame [px].
    bool update(const uint8_t* bgr, int width, int height, size_t stride, double& dx, double& dy, double& response);
    void reset() { have_prev_ = false; }

private:
    static constexpr int N = 256, SCALE = 4;
    std::vector<std::complex<float>> prev_, cur_;
    std::vector<float> window_;
    bool have_prev_ = false;
};

}  // namespace ledrx
