#include "motion.hpp"

#include <algorithm>
#include <cmath>
#include <future>

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
float sample_border(const T* src, int w, int h, int x, int y, bool replicate) {
    if (x < 0 || y < 0 || x >= w || y >= h) {
        if (!replicate) return 0.0f;
        x = std::clamp(x, 0, w - 1);
        y = std::clamp(y, 0, h - 1);
    }
    return static_cast<float>(src[static_cast<size_t>(y) * w + x]);
}

template <class T>
float bilinear_border(const T* src, int w, int h, double sx, double sy, bool replicate) {
    int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
    float fx = static_cast<float>(sx - x0), fy = static_cast<float>(sy - y0);
    float top = sample_border(src, w, h, x0, y0, replicate) * (1 - fx) + sample_border(src, w, h, x0 + 1, y0, replicate) * fx;
    float bot = sample_border(src, w, h, x0, y0 + 1, replicate) * (1 - fx) +
                sample_border(src, w, h, x0 + 1, y0 + 1, replicate) * fx;
    return top * (1 - fy) + bot * fy;
}

// dst(x, y) = src(x - tx, y - ty): every pixel has the same interpolation weights, rows are read in order.
template <class T>
void translate_impl(const T* src, float* dst, int w, int h, double tx, double ty, bool replicate) {
    const int ox = static_cast<int>(std::floor(-tx)), oy = static_cast<int>(std::floor(-ty));
    const float fx = static_cast<float>(-tx - ox), fy = static_cast<float>(-ty - oy);
    const float w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy), w10 = (1 - fx) * fy, w11 = fx * fy;
    const int x_lo = std::max(0, -ox), x_hi = std::min(w, w - 1 - ox);
    for (int y = 0; y < h; ++y) {
        float* out = dst + static_cast<size_t>(y) * w;
        const int sy = y + oy;
        if (sy < 0 || sy + 1 >= h || x_lo >= x_hi) {
            for (int x = 0; x < w; ++x) out[x] = bilinear_border(src, w, h, x - tx, y - ty, replicate);
            continue;
        }
        const T* a = src + static_cast<size_t>(sy) * w + ox;
        const T* b = a + w;
        for (int x = 0; x < x_lo; ++x) out[x] = bilinear_border(src, w, h, x - tx, y - ty, replicate);
        for (int x = x_lo; x < x_hi; ++x)
            out[x] = w00 * static_cast<float>(a[x]) + w01 * static_cast<float>(a[x + 1]) + w10 * static_cast<float>(b[x]) +
                     w11 * static_cast<float>(b[x + 1]);
        for (int x = x_hi; x < w; ++x) out[x] = bilinear_border(src, w, h, x - tx, y - ty, replicate);
    }
}

// Dark frames are mostly zero, so only 32x32 tiles whose source area holds something above `zero_below`
// are interpolated; the rest are zero-filled.
template <class T>
void warp_impl(const T* src, float* dst, int w, int h, const Homography& H, bool replicate, float zero_below) {
    const double* m = H.h;
    if (m[0] == 1 && m[1] == 0 && m[3] == 0 && m[4] == 1 && m[6] == 0 && m[7] == 0 && m[8] == 1) {
        translate_impl(src, dst, w, h, m[2], m[5], replicate);
        return;
    }
    constexpr int TS = 32;
    const int tw = (w + TS - 1) / TS, th = (h + TS - 1) / TS;
    std::vector<float> tile_max(static_cast<size_t>(tw) * th, 0.0f);
    auto max_rows = [&](int ta, int tb) {
        for (int y = ta * TS; y < std::min(h, tb * TS); ++y) {
            const T* row = src + static_cast<size_t>(y) * w;
            float* tm = tile_max.data() + static_cast<size_t>(y / TS) * tw;
            for (int t = 0; t < tw; ++t) {
                T mx = 0;
                for (int x = t * TS, e = std::min(w, x + TS); x < e; ++x) mx = std::max(mx, row[x]);
                tm[t] = std::max(tm[t], static_cast<float>(mx));
            }
        }
    };
    auto half = std::async(std::launch::async, max_rows, 0, th / 2);
    max_rows(th / 2, th);
    half.get();
    const Homography inv = H.inverse();
    auto tile_rows = [&](int ta, int tb) {
    for (int ty = ta; ty < tb; ++ty)
        for (int tx = 0; tx < tw; ++tx) {
            const int x0 = tx * TS, x1 = std::min(w, x0 + TS), y0 = ty * TS, y1 = std::min(h, y0 + TS);
            double lo_x = 1e30, hi_x = -1e30, lo_y = 1e30, hi_y = -1e30;
            for (int c = 0; c < 4; ++c) {
                double sx, sy;
                inv.apply(c & 1 ? x1 : x0, c & 2 ? y1 : y0, sx, sy);
                lo_x = std::min(lo_x, sx), hi_x = std::max(hi_x, sx), lo_y = std::min(lo_y, sy), hi_y = std::max(hi_y, sy);
            }
            int sx0 = std::clamp(static_cast<int>(std::floor(lo_x)) - 2, 0, w - 1);
            int sx1 = std::clamp(static_cast<int>(std::floor(hi_x)) + 2, 0, w - 1);
            int sy0 = std::clamp(static_cast<int>(std::floor(lo_y)) - 2, 0, h - 1);
            int sy1 = std::clamp(static_cast<int>(std::floor(hi_y)) + 2, 0, h - 1);
            float mx = 0;
            for (int a = sy0 / TS; a <= sy1 / TS; ++a)
                for (int b = sx0 / TS; b <= sx1 / TS; ++b) mx = std::max(mx, tile_max[static_cast<size_t>(a) * tw + b]);
            if (mx <= zero_below) {
                for (int y = y0; y < y1; ++y) std::fill(dst + static_cast<size_t>(y) * w + x0, dst + static_cast<size_t>(y) * w + x1, 0.0f);
                continue;
            }
            // Exact projective mapping at the tile corners, linear in between (the error is far below a
            // pixel for frame-to-frame rotations).
            double c[4][2];
            for (int k = 0; k < 4; ++k) inv.apply(k & 1 ? x1 : x0, k & 2 ? y1 : y0, c[k][0], c[k][1]);
            const double span_x = x1 - x0, span_y = y1 - y0;
            for (int y = y0; y < y1; ++y) {
                float* out = dst + static_cast<size_t>(y) * w;
                double v = (y - y0) / span_y;
                double lx = c[0][0] + (c[2][0] - c[0][0]) * v, ly = c[0][1] + (c[2][1] - c[0][1]) * v;
                double rx = c[1][0] + (c[3][0] - c[1][0]) * v, ry = c[1][1] + (c[3][1] - c[1][1]) * v;
                float sx = static_cast<float>(lx), sy = static_cast<float>(ly);
                const float dsx = static_cast<float>((rx - lx) / span_x), dsy = static_cast<float>((ry - ly) / span_x);
                for (int x = x0; x < x1; ++x, sx += dsx, sy += dsy) {
                    int ix = static_cast<int>(std::floor(sx)), iy = static_cast<int>(std::floor(sy));
                    if (ix < 0 || iy < 0 || ix >= w - 1 || iy >= h - 1) {
                        out[x] = bilinear_border(src, w, h, sx, sy, replicate);
                        continue;
                    }
                    float fx = sx - ix, fy = sy - iy;
                    const T* p = src + static_cast<size_t>(iy) * w + ix;
                    float top = p[0] + (static_cast<float>(p[1]) - p[0]) * fx;
                    float bot = p[w] + (static_cast<float>(p[w + 1]) - p[w]) * fx;
                    out[x] = top + (bot - top) * fy;
                }
            }
        }
    };
    auto first = std::async(std::launch::async, tile_rows, 0, th / 2);
    tile_rows(th / 2, th);
    first.get();
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

void warp(const float* src, float* dst, int w, int h, const Homography& H, bool replicate, float zero_below) {
    warp_impl(src, dst, w, h, H, replicate, zero_below);
}

void warp(const uint8_t* src, float* dst, int w, int h, const Homography& H, bool replicate, float zero_below) {
    warp_impl(src, dst, w, h, H, replicate, zero_below);
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
