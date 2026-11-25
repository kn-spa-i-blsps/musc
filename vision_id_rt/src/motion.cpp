#include "motion.hpp"

#include <algorithm>
#include <cmath>

namespace ledrx {
namespace {

using cf = std::complex<float>;

void fft(cf* a, int n, size_t step, bool inverse) {
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i * step], a[j * step]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI / len * (inverse ? 1 : -1);
        cf wl(static_cast<float>(std::cos(ang)), static_cast<float>(std::sin(ang)));
        for (int i = 0; i < n; i += len) {
            cf w(1, 0);
            for (int k = 0; k < len / 2; ++k) {
                cf u = a[(i + k) * step], v = a[(i + k + len / 2) * step] * w;
                a[(i + k) * step] = u + v;
                a[(i + k + len / 2) * step] = u - v;
                w *= wl;
            }
        }
    }
}

void fft2(std::vector<cf>& a, int n, bool inverse) {
    for (int r = 0; r < n; ++r) fft(a.data() + static_cast<size_t>(r) * n, n, 1, inverse);
    for (int c = 0; c < n; ++c) fft(a.data() + c, n, n, inverse);
}

template <class T>
void warp_impl(const T* src, float* dst, int w, int h, const Homography& H, bool replicate) {
    Homography inv = H.inverse();
    auto at = [&](int x, int y) -> float {
        if (x < 0 || y < 0 || x >= w || y >= h) {
            if (!replicate) return 0.0f;
            x = std::clamp(x, 0, w - 1);
            y = std::clamp(y, 0, h - 1);
        }
        return static_cast<float>(src[static_cast<size_t>(y) * w + x]);
    };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            double sx, sy;
            inv.apply(x, y, sx, sy);
            int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
            float fx = static_cast<float>(sx - x0), fy = static_cast<float>(sy - y0);
            float top = at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx;
            float bot = at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx;
            dst[static_cast<size_t>(y) * w + x] = top * (1 - fy) + bot * fy;
        }
}

}  // namespace

void Homography::apply(double x, double y, double& ox, double& oy) const {
    double z = h[6] * x + h[7] * y + h[8];
    ox = (h[0] * x + h[1] * y + h[2]) / z;
    oy = (h[3] * x + h[4] * y + h[5]) / z;
}

Homography Homography::inverse() const {
    const double* m = h;
    double c00 = m[4] * m[8] - m[5] * m[7], c01 = m[5] * m[6] - m[3] * m[8], c02 = m[3] * m[7] - m[4] * m[6];
    double det = m[0] * c00 + m[1] * c01 + m[2] * c02;
    return {{c00 / det, (m[2] * m[7] - m[1] * m[8]) / det, (m[1] * m[5] - m[2] * m[4]) / det,
             c01 / det, (m[0] * m[8] - m[2] * m[6]) / det, (m[2] * m[3] - m[0] * m[5]) / det,
             c02 / det, (m[1] * m[6] - m[0] * m[7]) / det, (m[0] * m[4] - m[1] * m[3]) / det}};
}

Homography Homography::translation(double dx, double dy) { return {{1, 0, dx, 0, 1, dy, 0, 0, 1}}; }

Homography rotation_homography(const double rot[3], int w, int h, double hfov_deg) {
    // Rodrigues: rotation vector -> R.
    double th = std::sqrt(rot[0] * rot[0] + rot[1] * rot[1] + rot[2] * rot[2]);
    double R[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    if (th > 1e-12) {
        double kx = rot[0] / th, ky = rot[1] / th, kz = rot[2] / th, c = std::cos(th), s = std::sin(th), v = 1 - c;
        R[0][0] = c + kx * kx * v;
        R[0][1] = kx * ky * v - kz * s;
        R[0][2] = kx * kz * v + ky * s;
        R[1][0] = ky * kx * v + kz * s;
        R[1][1] = c + ky * ky * v;
        R[1][2] = ky * kz * v - kx * s;
        R[2][0] = kz * kx * v - ky * s;
        R[2][1] = kz * ky * v + kx * s;
        R[2][2] = c + kz * kz * v;
    }
    double fx = (w / 2.0) / std::tan(hfov_deg * M_PI / 180.0 / 2), cx = w / 2.0, cy = h / 2.0;
    double K[3][3] = {{fx, 0, cx}, {0, fx, cy}, {0, 0, 1}};
    double Ki[3][3] = {{1 / fx, 0, -cx / fx}, {0, 1 / fx, -cy / fx}, {0, 0, 1}};
    double KRt[3][3] = {}, out[3][3] = {};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) KRt[i][j] += K[i][k] * R[j][k];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) out[i][j] += KRt[i][k] * Ki[k][j];
    Homography H;
    for (int i = 0; i < 9; ++i) H.h[i] = out[i / 3][i % 3];
    return H;
}

void warp(const float* src, float* dst, int w, int h, const Homography& H, bool replicate) {
    warp_impl(src, dst, w, h, H, replicate);
}

void warp(const uint8_t* src, float* dst, int w, int h, const Homography& H, bool replicate) {
    warp_impl(src, dst, w, h, H, replicate);
}

bool PhaseCorrelator::update(const uint8_t* bgr, int width, int height, size_t stride, double& dx, double& dy,
                             double& response) {
    const int sw = std::min(width / SCALE, N), sh = std::min(height / SCALE, N);
    const int ox = (width / SCALE - sw) / 2 * SCALE, oy = (height / SCALE - sh) / 2 * SCALE;
    if (window_.size() != static_cast<size_t>(N) * N) {
        window_.assign(static_cast<size_t>(N) * N, 0.0f);
        for (int y = 0; y < sh; ++y)
            for (int x = 0; x < sw; ++x)
                window_[static_cast<size_t>(y) * N + x] = static_cast<float>(
                    (0.5 - 0.5 * std::cos(2 * M_PI * x / (sw - 1))) * (0.5 - 0.5 * std::cos(2 * M_PI * y / (sh - 1))));
        have_prev_ = false;
    }
    cur_.assign(static_cast<size_t>(N) * N, cf(0, 0));
    for (int y = 0; y < sh; ++y)
        for (int x = 0; x < sw; ++x) {
            int sum = 0;
            for (int by = 0; by < SCALE; ++by) {
                const uint8_t* p = bgr + (oy + y * SCALE + by) * stride + 3 * (ox + x * SCALE);
                for (int bx = 0; bx < SCALE; ++bx, p += 3) sum += (p[2] * 4899 + p[1] * 9617 + p[0] * 1868 + 8192) >> 14;
            }
            size_t i = static_cast<size_t>(y) * N + x;
            cur_[i] = cf(window_[i] * sum / (SCALE * SCALE), 0);
        }
    fft2(cur_, N, false);
    bool ok = have_prev_;
    if (ok) {
        std::vector<cf> c(cur_.size());
        for (size_t i = 0; i < c.size(); ++i) {
            cf v = cur_[i] * std::conj(prev_[i]);
            float mag = std::abs(v);
            c[i] = mag > 1e-12f ? v / mag : cf(0, 0);
        }
        fft2(c, N, true);
        const float norm = 1.0f / (static_cast<float>(N) * N);
        size_t best = 0;
        for (size_t i = 1; i < c.size(); ++i)
            if (c[i].real() > c[best].real()) best = i;
        int py = static_cast<int>(best / N), px = static_cast<int>(best % N);
        double sum = 0, sx = 0, sy = 0;
        for (int ky = -2; ky <= 2; ++ky)
            for (int kx = -2; kx <= 2; ++kx) {
                float v = c[static_cast<size_t>((py + ky + N) % N) * N + (px + kx + N) % N].real() * norm;
                sum += v;
                sx += v * kx;
                sy += v * ky;
            }
        double cxp = px + (sum != 0 ? sx / sum : 0), cyp = py + (sum != 0 ? sy / sum : 0);
        if (cxp > N / 2.0) cxp -= N;
        if (cyp > N / 2.0) cyp -= N;
        dx = cxp * SCALE;
        dy = cyp * SCALE;
        response = sum;
    }
    prev_.swap(cur_);
    have_prev_ = true;
    return ok;
}

}  // namespace ledrx
