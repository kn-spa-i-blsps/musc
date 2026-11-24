// Per-row image kernels, NEON with a scalar fallback.
#pragma once
#include <cstdint>
#include <cstdlib>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace ledrx {

// max(R, G) - B, saturated at 0 (nonblue LED map).
inline void nonblue_row(const uint8_t* bgr, uint8_t* out, int w) {
    int x = 0;
#if defined(__ARM_NEON)
    for (; x + 16 <= w; x += 16) {
        uint8x16x3_t p = vld3q_u8(bgr + 3 * x);
        vst1q_u8(out + x, vqsubq_u8(vmaxq_u8(p.val[2], p.val[1]), p.val[0]));
    }
#endif
    for (; x < w; ++x) {
        int b = bgr[3 * x], g = bgr[3 * x + 1], r = bgr[3 * x + 2];
        int v = (r > g ? r : g) - b;
        out[x] = static_cast<uint8_t>(v > 0 ? v : 0);
    }
}

inline int reflect101(int i, int n) {
    if (i < 0) return -i;
    if (i >= n) return 2 * n - 2 - i;
    return i;
}

// |cur - prev| blurred horizontally with [1 4 6 4 1] (unnormalized, max 4080).
inline void absdiff_hblur_row(const uint8_t* cur, const uint8_t* prev, uint8_t* d, uint16_t* out, int w) {
    int x = 0;
#if defined(__ARM_NEON)
    for (; x + 16 <= w; x += 16) vst1q_u8(d + x, vabdq_u8(vld1q_u8(cur + x), vld1q_u8(prev + x)));
#endif
    for (; x < w; ++x) d[x] = static_cast<uint8_t>(std::abs(cur[x] - prev[x]));

    auto tap = [&](int x) {
        return d[reflect101(x - 2, w)] + 4 * d[reflect101(x - 1, w)] + 6 * d[x] + 4 * d[reflect101(x + 1, w)] +
               d[reflect101(x + 2, w)];
    };
    out[0] = tap(0);
    out[1] = tap(1);
    x = 2;
#if defined(__ARM_NEON)
    const uint8x8_t four = vdup_n_u8(4), six = vdup_n_u8(6);
    for (; x + 8 <= w - 2; x += 8) {
        uint16x8_t s = vaddl_u8(vld1_u8(d + x - 2), vld1_u8(d + x + 2));
        s = vmlal_u8(s, vld1_u8(d + x - 1), four);
        s = vmlal_u8(s, vld1_u8(d + x + 1), four);
        s = vmlal_u8(s, vld1_u8(d + x), six);
        vst1q_u16(out + x, s);
    }
#endif
    for (; x < w - 2; ++x) out[x] = d[x - 2] + 4 * d[x - 1] + 6 * d[x] + 4 * d[x + 1] + d[x + 2];
    out[w - 2] = tap(w - 2);
    out[w - 1] = tap(w - 1);
}

// Vertical [1 4 6 4 1] over five hblur rows, /256, then energy = max(blur, energy * decay).
// Appends indices (row_offset + x) of pixels with energy >= thr.
inline void vblur_energy_row(const uint16_t* const r[5], float* en, int w, bool decay_old, float decay, float thr,
                             uint32_t row_offset, std::vector<uint32_t>& hits) {
    const float scale = 1.0f / 256;
    int x = 0;
#if defined(__ARM_NEON)
    const float32x4_t vscale = vdupq_n_f32(scale), vdecay = vdupq_n_f32(decay), vthr = vdupq_n_f32(thr);
    for (; x + 8 <= w; x += 8) {
        uint16x8_t a = vld1q_u16(r[0] + x), e4 = vld1q_u16(r[4] + x);
        uint16x8_t s = vaddq_u16(a, e4);
        s = vmlaq_n_u16(s, vaddq_u16(vld1q_u16(r[1] + x), vld1q_u16(r[3] + x)), 4);
        s = vmlaq_n_u16(s, vld1q_u16(r[2] + x), 6);
        float32x4_t lo = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(s))), vscale);
        float32x4_t hi = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(s))), vscale);
        if (decay_old) {
            lo = vmaxq_f32(lo, vmulq_f32(vld1q_f32(en + x), vdecay));
            hi = vmaxq_f32(hi, vmulq_f32(vld1q_f32(en + x + 4), vdecay));
        }
        vst1q_f32(en + x, lo);
        vst1q_f32(en + x + 4, hi);
        uint32x4_t any = vorrq_u32(vcgeq_f32(lo, vthr), vcgeq_f32(hi, vthr));
        if (vmaxvq_u32(any))
            for (int k = x; k < x + 8; ++k)
                if (en[k] >= thr) hits.push_back(row_offset + k);
    }
#endif
    for (; x < w; ++x) {
        float v = static_cast<float>(r[0][x] + 4 * r[1][x] + 6 * r[2][x] + 4 * r[3][x] + r[4][x]) * scale;
        en[x] = decay_old ? (v > en[x] * decay ? v : en[x] * decay) : v;
        if (en[x] >= thr) hits.push_back(row_offset + x);
    }
}

// |cur - prev * gain| blurred horizontally with [1 4 6 4 1] / 16.
inline void scaled_absdiff_hblur_row(const uint8_t* cur, const uint8_t* prev, float gain, float* d, float* out,
                                     int w) {
    for (int x = 0; x < w; ++x) {
        float v = cur[x] - prev[x] * gain;
        d[x] = v < 0 ? -v : v;
    }
    auto tap = [&](int x) {
        return (d[reflect101(x - 2, w)] + 4 * d[reflect101(x - 1, w)] + 6 * d[x] + 4 * d[reflect101(x + 1, w)] +
                d[reflect101(x + 2, w)]) * (1.0f / 16);
    };
    out[0] = tap(0);
    out[1] = tap(1);
    for (int x = 2; x < w - 2; ++x) out[x] = (d[x - 2] + 4 * d[x - 1] + 6 * d[x] + 4 * d[x + 1] + d[x + 2]) * (1.0f / 16);
    out[w - 2] = tap(w - 2);
    out[w - 1] = tap(w - 1);
}

inline void scaled_vblur_energy_row(const float* const r[5], float* en, int w, bool decay_old, float decay, float thr,
                                    uint32_t row_offset, std::vector<uint32_t>& hits) {
    for (int x = 0; x < w; ++x) {
        float v = (r[0][x] + 4 * r[1][x] + 6 * r[2][x] + 4 * r[3][x] + r[4][x]) * (1.0f / 16);
        float old = en[x] * decay;
        en[x] = decay_old && old > v ? old : v;
    }
    for (int x = 0; x < w; ++x)
        if (en[x] >= thr) hits.push_back(row_offset + x);
}

}  // namespace ledrx
