#include "decoder.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "codebook.h"
#include "stats.hpp"

namespace ledrx {
namespace {

constexpr int PHASE_STEPS = 4;
constexpr double RATES[] = {0.993, 1.0, 1.007};
constexpr double EDGE_ZONE = 0.2;
constexpr float EDGE_WEIGHT = 0.3f;
constexpr double MIN_SCORE = 0.8, MIN_MARGIN = 0.06, MIN_COVERAGE = 0.8;
constexpr int N_ROT = N_IDS * CHIPS;

// A phase shift by a whole chip only rotates the chip bins, so every (id, rotation) score is a
// dot product of one row of this table with the binned samples: rot[c][id * CHIPS + q].
struct RotationTable {
    std::array<std::array<float, N_ROT>, CHIPS> rot;
    RotationTable() {
        for (int k = 0; k < N_IDS; ++k)
            for (int q = 0; q < CHIPS; ++q)
                for (int c = 0; c < CHIPS; ++c) {
                    int chip = (c + q) % CHIPS;
                    int bit = (CODES[k] >> (CHIPS - 1 - chip)) & 1;
                    rot[c][k * CHIPS + q] = bit ? 1.0f : -1.0f;
                }
    }
};

const RotationTable& rotation_table() {
    static const RotationTable table;
    return table;
}

}  // namespace

TemplateDecoder::TemplateDecoder(double chip_hz, double min_contrast)
    : chip_ms(1000.0 / chip_hz), frame_ms(CHIPS * 1000.0 / chip_hz), min_contrast_(min_contrast) {}

int TemplateDecoder::push(double ts_ms, double value) {
    samples_.emplace_back(ts_ms, static_cast<float>(value));
    while (!samples_.empty() && ts_ms - samples_.front().first > 1.2 * frame_ms) samples_.pop_front();
    coverage = std::min(1.0, (ts_ms - samples_.front().first) / frame_ms);
    ++n_since_eval_;
    if (coverage < MIN_COVERAGE || n_since_eval_ < 2) return NO_ID;
    n_since_eval_ = 0;
    return evaluate(ts_ms);
}

bool TemplateDecoder::normalize() {
    sorted_.assign(v_.begin(), v_.end());
    double lo, hi;
    percentiles(sorted_, 10, 90, lo, hi);
    if (hi - lo <= std::max(min_contrast_, 1e-6)) return false;
    float mid = static_cast<float>(0.5 * (lo + hi)), half = static_cast<float>(0.5 * (hi - lo));
    s_.resize(v_.size());
    for (size_t i = 0; i < v_.size(); ++i) s_[i] = std::clamp((v_[i] - mid) / half, -1.0f, 1.0f);
    return true;
}

bool TemplateDecoder::looks_like_beacon() const {
    size_t n = s_.size();
    size_t dark = std::count_if(s_.begin(), s_.end(), [](float s) { return s < 0; });
    double duty = 1.0 - static_cast<double>(dark) / n;
    if (duty < 0.5 || duty > 0.92) return false;
    double dt = 0.0;
    if (n > 1) {
        std::vector<double> diffs(n - 1);
        for (size_t i = 1; i < n; ++i) diffs[i - 1] = t_[i] - t_[i - 1];
        dt = median(diffs);
    }
    double max_dark_ms = (MAX_DARK + 0.6) * chip_ms + dt;
    bool in_dark = false;
    double start = 0.0;
    for (size_t i = 0; i < n; ++i) {
        bool d = s_[i] < 0;
        if (d && !in_dark) {
            in_dark = true;
            start = t_[i];
        } else if (!d && in_dark) {
            if (t_[i] - start > max_dark_ms) return false;
            in_dark = false;
        }
    }
    return true;
}

bool TemplateDecoder::id_scores(double now, float* per_id) {
    size_t n = samples_.size();
    t_.resize(n);
    v_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        t_[i] = samples_[i].first - now;
        v_[i] = samples_[i].second;
    }
    if (!normalize() || !looks_like_beacon()) return false;

    const auto& rot = rotation_table().rot;
    std::fill(per_id, per_id + N_IDS, -INFINITY);
    alignas(64) float acc[N_ROT];
    for (double rate : RATES) {
        for (int sub = 0; sub < PHASE_STEPS; ++sub) {
            double phase = sub * (chip_ms / PHASE_STEPS);
            float bins[CHIPS] = {};
            float norm = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                double pos = (t_[i] * rate + phase) / chip_ms;
                double fl = std::floor(pos);
                double frac = pos - fl;
                int chip = static_cast<int>(static_cast<long long>(fl) % CHIPS);
                if (chip < 0) chip += CHIPS;
                float w = (frac > EDGE_ZONE && frac < 1 - EDGE_ZONE) ? 1.0f : EDGE_WEIGHT;
                bins[chip] += w * s_[i];
                norm += w * std::fabs(s_[i]);
            }
            if (norm <= 0.0f) continue;
            std::fill(acc, acc + N_ROT, 0.0f);
            for (int c = 0; c < CHIPS; ++c) {
                float b = bins[c];
                const float* row = rot[c].data();
                for (int j = 0; j < N_ROT; ++j) acc[j] += b * row[j];
            }
            float inv = 1.0f / norm;
            for (int k = 0; k < N_IDS; ++k) {
                float m = *std::max_element(acc + k * CHIPS, acc + (k + 1) * CHIPS) * inv;
                per_id[k] = std::max(per_id[k], m);
            }
        }
    }
    return true;
}

int TemplateDecoder::evaluate(double now) {
    float per_id[N_IDS];
    if (!id_scores(now, per_id)) {
        last_score = last_margin = 0.0;
        return NO_ID;
    }
    int best_k = static_cast<int>(std::max_element(per_id, per_id + N_IDS) - per_id);
    float best = per_id[best_k], second = -1.0f;
    for (int k = 0; k < N_IDS; ++k)
        if (k != best_k) second = std::max(second, per_id[k]);
    last_score = best;
    last_margin = best - second;

    bool full = coverage >= 0.95;
    double margin_needed = MIN_MARGIN * (full ? 1.0 : 1.5);
    double score_needed = full ? MIN_SCORE : std::max(MIN_SCORE, 0.9);
    if (best < score_needed || best - second < margin_needed) {
        if (full) ++frames_bad;
        return NO_ID;
    }
    int id = best_k;
    if (id != last_id) {
        cand_since_ = now;
        if (confirmed_id != NO_ID && confirmed_id != id) confirmed_id = NO_ID;
    } else if (now - cand_since_ >= 0.5 * frame_ms) {
        if (confirmed_id != id) confirmed_ts = now;
        confirmed_id = id;
    }
    if (id != last_id || now - last_ok_ts > 0.5 * frame_ms) {
        ++frames_ok;
        last_ok_ts = now;
    }
    last_id = id;
    return id;
}

}  // namespace ledrx
