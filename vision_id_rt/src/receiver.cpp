#include "receiver.hpp"

#include <algorithm>
#include <cmath>
#include <future>
#include <set>
#include <tuple>

#include "codebook.h"
#include "kernels.hpp"
#include "stats.hpp"

namespace ledrx {
namespace {

constexpr size_t LEVELS_LEN = 90;
constexpr size_t MAX_PEAKS = 60;
constexpr double LOST_KEEP_MS = 800.0;

template <class T>
void push_capped(std::deque<T>& d, T v, size_t cap) {
    d.push_back(v);
    if (d.size() > cap) d.pop_front();
}

struct Window {
    int x0, y0, x1, y1;
    bool too_small(int min_side) const { return x1 - x0 < min_side || y1 - y0 < min_side; }
};

Window window(int cx, int cy, int r, int w, int h) {
    return {std::max(cx - r, 0), std::max(cy - r, 0), std::min(cx + r + 1, w), std::min(cy + r + 1, h)};
}

}  // namespace

Receiver::Receiver(Config cfg)
    : cfg_(cfg), grouper_(CHIPS * cfg.chip_ms(), cfg.chip_ms(), cfg.cs_min_events, cfg.group_radius_px, 0.5,
                          cfg.cs_near_weight) {}

// ------------------------------------------------------------------ camera motion

void Receiver::warp_history(const Homography& H) {
    const size_t n = static_cast<size_t>(w_) * h_;
    // Energy never grows by itself and every decision starts at 15, so values below 1 can only fade out;
    // zeroing them while warping changes no decision and leaves most tiles empty.
    constexpr float ENERGY_FLOOR = 1.0f;
    std::future<void> energy_done;
    if (have_energy_) {
        energy_tmp_.resize(n);
        energy_done = std::async(std::launch::async, [&] {
            warp(energy_.data(), energy_tmp_.data(), w_, h_, H, false, ENERGY_FLOOR);
        });
    }
    if (have_feat_) {
        prev_warped_.resize(n);
        if (have_warped_) {
            scratch_.resize(n);
            warp(prev_warped_.data(), scratch_.data(), w_, h_, H, true);
            prev_warped_.swap(scratch_);
        } else {
            warp(feat_.data(), prev_warped_.data(), w_, h_, H, true);
        }
        have_warped_ = true;
    }
    if (energy_done.valid()) {
        energy_done.get();
        energy_.swap(energy_tmp_);
    }
}

// The gyro gives the camera rotation directly, so this works on a plain sky where the image has nothing to lock on.
void Receiver::compensate_rotation(const double rot[3]) {
    Homography H = rotation_homography(rot, w_, h_, cfg_.hfov_deg);
    const double pts[5][2] = {{0, 0}, {double(w_), 0}, {0, double(h_)}, {double(w_), double(h_)}, {w_ / 2.0, h_ / 2.0}};
    double moved = 0;
    for (const auto& p : pts) {
        double x, y;
        H.apply(p[0], p[1], x, y);
        moved = std::max({moved, std::fabs(x - p[0]), std::fabs(y - p[1])});
    }
    double cx, cy;
    H.apply(w_ / 2.0, h_ / 2.0, cx, cy);
    motion_[0] = cx - w_ / 2.0;
    motion_[1] = cy - h_ / 2.0;
    if (moved < 0.3) {
        motion_[0] = motion_[1] = 0;
        return;
    }
    warp_history(H);
    phase_.reset();
    for (auto& t : tracks_) {
        double x, y;
        H.apply(t->kf.x[0], t->kf.x[1], x, y);
        t->kf.shift(x - t->kf.x[0], y - t->kf.x[1]);
        t->x = t->kf.x[0];
        t->y = t->kf.x[1];
    }
    for (Lost& l : lost_) H.apply(l.x, l.y, l.x, l.y);
}

void Receiver::compensate_shift(const uint8_t* bgr, size_t stride) {
    motion_[0] = motion_[1] = 0;
    double dx, dy, response;
    if (!phase_.update(bgr, w_, h_, stride, dx, dy, response)) return;
    if (response < 0.2 || std::fabs(dx) > w_ / 10.0 || std::fabs(dy) > h_ / 10.0 ||
        (std::fabs(dx) < 0.5 && std::fabs(dy) < 0.5))
        return;
    motion_[0] = dx;
    motion_[1] = dy;
    warp_history(Homography::translation(dx, dy));
    for (auto& t : tracks_) {
        t->kf.shift(dx, dy);
        t->x = t->kf.x[0];
        t->y = t->kf.x[1];
    }
}

// ------------------------------------------------------------------ blink energy

void Receiver::update_feature_and_energy(const uint8_t* bgr, size_t stride) {
    const int w = w_, h = h_;
    if (!have_feat_) {
        for (int y = 0; y < h; ++y) nonblue_row(bgr + y * stride, feat_.data() + static_cast<size_t>(y) * w, w);
        have_feat_ = true;
        return;
    }
    prev_feat_.swap(feat_);
    candidates_.clear();
    const float gain = exposure_gain(bgr, stride);
    const bool exact = gain == 1.0f && !have_warped_;

    // One pass over rows: feature and horizontal blur run two rows ahead of the vertical blur.
    auto slot = [&](int y) { return static_cast<size_t>(y % HBLUR_ROWS) * w; };
    auto make_row = [&](int y) {
        uint8_t* cur = feat_.data() + static_cast<size_t>(y) * w;
        const uint8_t* prev = prev_feat_.data() + static_cast<size_t>(y) * w;
        nonblue_row(bgr + y * stride, cur, w);
        if (exact)
            absdiff_hblur_row(cur, prev, diff_.data(), hblur_.data() + slot(y), w);
        else if (have_warped_)
            scaled_absdiff_hblur_row(cur, prev_warped_.data() + static_cast<size_t>(y) * w, gain, fdiff_.data(),
                                     fhblur_.data() + slot(y), w);
        else
            scaled_absdiff_hblur_row(cur, prev, gain, fdiff_.data(), fhblur_.data() + slot(y), w);
    };
    make_row(0);
    make_row(1);
    for (int y = 0; y < h; ++y) {
        if (y + 2 < h) make_row(y + 2);
        float* en = energy_.data() + static_cast<size_t>(y) * w;
        uint32_t offset = static_cast<uint32_t>(y) * w;
        if (exact) {
            const uint16_t* rows[5];
            for (int k = 0; k < 5; ++k) rows[k] = hblur_.data() + slot(reflect101(y + k - 2, h));
            vblur_energy_row(rows, en, w, have_energy_, cfg_.energy_decay, cfg_.energy_threshold, offset, candidates_);
        } else {
            const float* rows[5];
            for (int k = 0; k < 5; ++k) rows[k] = fhblur_.data() + slot(reflect101(y + k - 2, h));
            scaled_vblur_energy_row(rows, en, w, have_energy_, cfg_.energy_decay, cfg_.energy_threshold, offset,
                                    candidates_);
        }
    }
    have_energy_ = true;
    have_warped_ = false;
}

// Auto exposure changes the whole image at once; without compensation every edge would blink.
float Receiver::exposure_gain(const uint8_t* bgr, size_t stride) const {
    std::vector<float> ratios;
    for (int y = 0; y < h_; y += 8) {
        const uint8_t* row = bgr + y * stride;
        for (int x = 0; x < w_; x += 8) {
            int b = row[3 * x], g = row[3 * x + 1], r = row[3 * x + 2];
            float c = static_cast<float>(std::max(std::max(r, g) - b, 0));
            size_t i = static_cast<size_t>(y) * w_ + x;
            float p = have_warped_ ? prev_warped_[i] : prev_feat_[i];
            if (c > 20 && c < 235 && p > 20 && p < 235) ratios.push_back(c / p);
        }
    }
    if (ratios.size() <= 100) return 1.0f;
    return static_cast<float>(median(ratios));
}

void Receiver::find_peaks() {
    peaks_.clear();
    if (!have_energy_) return;
    const int w = w_, h = h_, k = cfg_.peak_nms_px / 2;
    std::vector<float> ring;
    for (uint32_t idx : candidates_) {
        int x = idx % w, y = idx / w;
        float v = e(x, y);
        bool is_max = true;
        Window nb = window(x, y, k, w, h);
        for (int yy = nb.y0; yy < nb.y1 && is_max; ++yy)
            for (int xx = nb.x0; xx < nb.x1; ++xx)
                if (e(xx, yy) > v) {
                    is_max = false;
                    break;
                }
        if (!is_max) continue;

        // A compact spot, not a line (edge during motion): energy drops on a ring around the peak.
        ring.clear();
        for (int yy = std::max(y - 14, 0); yy < std::min(y + 15, h); yy += 7)
            for (int xx = std::max(x - 14, 0); xx < std::min(x + 15, w); xx += 7) ring.push_back(e(xx, yy));
        if (median(ring) > 0.5 * v) continue;

        Window c = window(x, y, 3, w, h);
        double wsum = 0, sx = 0, sy = 0;
        for (int yy = c.y0; yy < c.y1; ++yy)
            for (int xx = c.x0; xx < c.x1; ++xx) {
                float ev = e(xx, yy);
                wsum += ev;
                sx += xx * static_cast<double>(ev);
                sy += yy * static_cast<double>(ev);
            }
        Window a = window(x, y, 12, w, h);
        int area = 0;
        for (int yy = a.y0; yy < a.y1; ++yy)
            for (int xx = a.x0; xx < a.x1; ++xx) area += e(xx, yy) >= 0.5 * v;
        peaks_.push_back({sx / wsum, sy / wsum, std::sqrt(area / M_PI), v});
    }
    std::stable_sort(peaks_.begin(), peaks_.end(), [](const Peak& a, const Peak& b) { return a.energy > b.energy; });
    if (peaks_.size() > MAX_PEAKS) peaks_.resize(MAX_PEAKS);
}

// ------------------------------------------------------------------ tracking

// The LED is dark for about one chip; without prediction the track would look for it where it went dark.
void Receiver::predict(double ts, double dt) {
    for (auto& t : tracks_) {
        if (ts - t->last_seen_ts >= cfg_.kf_coast_ms) t->kf.stop();
        t->kf.predict(dt);
        t->x = t->kf.x[0];
        t->y = t->kf.x[1];
    }
}

void Receiver::correct(Track& t, double zx, double zy, double meas_std) {
    t.kf.correct(zx, zy, meas_std);
    t.kf.clamp_speed(cfg_.max_px_per_ms);
    t.x = t.kf.x[0];
    t.y = t.kf.x[1];
}

double Receiver::reach(const Track& t, double ts) const {
    double grow = std::max(cfg_.search_grow * (ts - t.last_seen_ts), cfg_.kf_gate_sigma * t.kf.pos_std());
    return cfg_.snap_px + t.radius + std::min(grow, 60.0);
}

void Receiver::assign_and_spawn(double ts) {
    std::vector<bool> used(peaks_.size(), false);
    const int nt = static_cast<int>(tracks_.size()), np = static_cast<int>(peaks_.size());
    if (nt && np) {
        // Cost: (distance from the prediction / gate)^2 minus a bonus for stronger peaks; outside the gate forbidden.
        constexpr double BIG = 1e6;
        double emax = 0;
        for (const Peak& p : peaks_) emax = std::max(emax, p.energy);
        std::vector<double> cost(static_cast<size_t>(nt) * np);
        std::vector<char> gate(cost.size());
        bool any = false;
        for (int i = 0; i < nt; ++i) {
            const Track& t = *tracks_[i];
            double r = reach(t, ts);
            for (int j = 0; j < np; ++j) {
                double d = std::hypot(t.x - peaks_[j].x, t.y - peaks_[j].y);
                size_t k = static_cast<size_t>(i) * np + j;
                gate[k] = d <= r;
                any |= gate[k] != 0;
                cost[k] = gate[k] ? (d / r) * (d / r) - cfg_.assign_energy_w * (peaks_[j].energy / emax) : BIG;
            }
        }
        if (any)
            for (auto [ti, pi] : hungarian(cost, nt, np)) {
                if (!gate[static_cast<size_t>(ti) * np + pi]) continue;
                used[pi] = true;
                Track& t = *tracks_[ti];
                const Peak& p = peaks_[pi];
                correct(t, p.x, p.y, cfg_.kf_peak_px);
                t.radius = 0.9 * t.radius + 0.1 * std::max(p.radius, 1.0);
                t.last_seen_ts = ts;
            }
    }

    for (size_t i = 0; i < peaks_.size(); ++i) {
        const Peak& p = peaks_[i];
        if (used[i] || p.energy < cfg_.spawn_threshold) continue;
        bool crowded = std::any_of(tracks_.begin(), tracks_.end(), [&](const auto& t) {
            return std::hypot(p.x - t->x, p.y - t->y) < cfg_.min_track_sep_px;
        });
        if (crowded) continue;
        if (tracks_.size() >= cfg_.max_tracks) {
            auto victim = tracks_.end();
            for (auto it = tracks_.begin(); it != tracks_.end(); ++it) {
                const Track& t = **it;
                if (t.drone_id != NO_ID || t.decoder.frames_ok != 0) continue;
                if (victim == tracks_.end() || t.contrast < (*victim)->contrast ||
                    (t.contrast == (*victim)->contrast && t.created_ts > (*victim)->created_ts))
                    victim = it;
            }
            if (victim == tracks_.end()) break;
            tracks_.erase(victim);
        }
        auto t = std::make_unique<Track>(next_tid_++, p.x, p.y, ts, cfg_, std::max(p.radius, 2.0));
        inherit_id(*t, ts);
        tracks_.push_back(std::move(t));
    }
}

void Receiver::inherit_id(Track& t, double ts) {
    std::set<int> live;
    for (auto& k : tracks_)
        if (k->drone_id != NO_ID) live.insert(k->drone_id);
    double best_d = INFINITY;
    int best_id = NO_ID;
    for (const Lost& l : lost_) {
        double d = std::hypot(t.x - l.x, t.y - l.y);
        if (!live.count(l.id) && ts - l.ts < LOST_KEEP_MS && d < 60.0 && d < best_d) {
            best_d = d;
            best_id = l.id;
        }
    }
    if (best_id != NO_ID) {
        t.drone_id = best_id;
        t.id_confirmed = false;
        t.inherited_ts = ts;
    }
}

// A reflection is always dimmer than the LED lighting it, so move to the brightest blinking point nearby.
void Receiver::seek_source(Track& t) {
    if (!have_energy_ || !t.has_lohi || t.states.empty() || !t.states.back()) return;
    int cx = py_round(t.x), cy = py_round(t.y);
    Window win = window(cx, cy, cfg_.source_search_px, w_, h_);
    if (win.too_small(3)) return;
    float emax = 0;
    for (int y = win.y0; y < win.y1; ++y)
        for (int x = win.x0; x < win.x1; ++x) emax = std::max(emax, e(x, y));
    if (emax < cfg_.energy_threshold) return;
    float blink_thr = std::max(cfg_.energy_threshold, 0.6f * emax);
    int peak = 0;
    for (int y = win.y0; y < win.y1; ++y)
        for (int x = win.x0; x < win.x1; ++x)
            if (e(x, y) >= blink_thr) peak = std::max(peak, static_cast<int>(f(x, y)));

    Window own_win = window(cx, cy, 4, w_, h_);
    int own = 0;
    for (int y = own_win.y0; y < own_win.y1; ++y)
        for (int x = own_win.x0; x < own_win.x1; ++x) own = std::max(own, static_cast<int>(f(x, y)));
    t.on_peak = t.on_peak != 0.0 ? 0.8 * t.on_peak + 0.2 * own : own;
    if (peak < own + cfg_.source_margin) return;

    double sx = 0, sy = 0;
    int n = 0;
    for (int y = win.y0; y < win.y1; ++y)
        for (int x = win.x0; x < win.x1; ++x)
            if (e(x, y) >= blink_thr && f(x, y) >= peak - 4) {
                sx += x;
                sy += y;
                ++n;
            }
    t.x = sx / n;
    t.y = sy / n;
}

// The LED is a bright point in the feature map; snap to it near the predicted position.
void Receiver::lock_point(Track& t, double ts) {
    int r = static_cast<int>(cfg_.snap_px + t.radius + std::min(cfg_.search_grow * (ts - t.last_seen_ts), 60.0));
    Window win = window(py_round(t.x), py_round(t.y), r, w_, h_);
    if (win.too_small(3)) return;
    int m = 0;
    for (int y = win.y0; y < win.y1; ++y)
        for (int x = win.x0; x < win.x1; ++x) m = std::max(m, static_cast<int>(f(x, y)));
    if (m < std::max(cfg_.point_min, cfg_.point_rel * t.point_peak)) {
        t.point_peak *= 0.995;
        return;
    }
    t.point_peak = t.point_peak == 0.0 ? m : 0.9 * t.point_peak + 0.1 * m;
    double thr = 0.7 * m, wsum = 0, sx = 0, sy = 0;
    for (int y = win.y0; y < win.y1; ++y)
        for (int x = win.x0; x < win.x1; ++x) {
            uint8_t v = f(x, y);
            if (v >= thr) {
                wsum += v;
                sx += static_cast<double>(x - win.x0) * v;
                sy += static_cast<double>(y - win.y0) * v;
            }
        }
    t.x = win.x0 + sx / wsum;
    t.y = win.y0 + sy / wsum;
    t.last_seen_ts = ts;
}

double Receiver::sample(Track& t) {
    int r = std::min(std::max(py_round(t.radius) + cfg_.sample_pad, 2), cfg_.sample_max);
    Window win = window(py_round(t.x), py_round(t.y), r, w_, h_);
    t.roi[0] = win.x0;
    t.roi[1] = win.y0;
    t.roi[2] = win.x1;
    t.roi[3] = win.y1;
    double level = 0.0;
    if (win.x1 > win.x0 && win.y1 > win.y0) {
        uint64_t sum = 0;
        for (int y = win.y0; y < win.y1; ++y)
            for (int x = win.x0; x < win.x1; ++x) sum += f(x, y);
        level = static_cast<double>(sum) / ((win.x1 - win.x0) * (win.y1 - win.y0));
    }
    push_capped(t.levels, level, LEVELS_LEN);
    // Young tracks update the threshold every frame, so the first dark chip already gives a state.
    size_t n = t.levels.size();
    if (n >= 6 && (n % 3 == 0 || n < 30 || !t.has_lohi)) {
        std::vector<double> v(t.levels.begin(), t.levels.end());
        percentiles(v, 10, 90, t.lo, t.hi);
        t.has_lohi = true;
        t.contrast = t.hi - t.lo;
    }
    return level;
}

void Receiver::update_id(Track& t, double ts) {
    const double frame_ms = CHIPS * cfg_.chip_ms();
    TemplateDecoder& dec = t.decoder;
    bool fresh = ts - dec.last_ok_ts < 1.5 * frame_ms;
    if (cfg_.constellations) {
        t.synced_with = 0;
    } else if (t.synced_with) {
        if (dec.last_id != NO_ID && fresh && dec.last_id != t.drone_id)
            t.synced_with = 0;
        else
            return;
    }
    if (!fresh) dec.confirmed_id = NO_ID;
    if (dec.confirmed_id != NO_ID && dec.last_id == dec.confirmed_id) {
        t.drone_id = dec.confirmed_id;
        t.id_confirmed = true;
        t.inherited_ts = NAN;
    } else if (dec.last_id != NO_ID && fresh) {
        t.drone_id = dec.last_id;
        t.id_confirmed = false;
    } else if (t.inherited() && ts - t.inherited_ts < 1.3 * frame_ms) {
    } else {
        t.drone_id = NO_ID;
        t.id_confirmed = false;
        t.inherited_ts = NAN;
    }
}

// Groups tracks by blinking (BlinkGrouper) and shares IDs inside a group: a track without its own confirmed
// read takes the group's ID from the other tracks' reads.
void Receiver::constellate(double ts) {
    const double frame_ms = CHIPS * cfg_.chip_ms();
    grouper_.radius_px = cfg_.group_radius_px;
    std::vector<int> tids, states;
    std::vector<double> xy;
    for (auto& t : tracks_) {
        tids.push_back(t->tid);
        xy.push_back(t->x);
        xy.push_back(t->y);
        states.push_back(t->cs_state);
    }
    std::map<int, int> gids = grouper_.update(ts, tids, xy, states);

    std::vector<int> order;
    std::map<int, std::vector<Track*>> groups;
    for (auto& t : tracks_) {
        t->gid = gids[t->tid];
        if (!groups.count(t->gid)) order.push_back(t->gid);
        groups[t->gid].push_back(t.get());
    }
    for (auto it = group_since_.begin(); it != group_since_.end();) {
        auto g = groups.find(it->first);
        it = g != groups.end() && g->second.size() >= 2 ? std::next(it) : group_since_.erase(it);
    }
    constellations_.clear();
    auto own = [](const Track* t) {
        const TemplateDecoder& d = t->decoder;
        return d.confirmed_id != NO_ID && d.last_id == d.confirmed_id ? d.confirmed_id : NO_ID;
    };
    for (int gid : order) {
        auto& members = groups[gid];
        if (members.size() < 2) continue;
        double since = group_since_.emplace(gid, ts).first->second;

        // Two tracks that confirmed different IDs are not one constellation.
        std::vector<std::pair<Track*, int>> conf;
        for (Track* t : members)
            if (own(t) != NO_ID) conf.emplace_back(t, own(t));
        for (size_t a = 0; a < conf.size(); ++a)
            for (size_t b = a + 1; b < conf.size(); ++b)
                if (conf[a].second != conf[b].second) grouper_.block(conf[a].first->tid, conf[b].first->tid);

        std::map<int, int> votes;
        for (auto& c : conf) ++votes[c.second];
        bool confirmed = !votes.empty();
        if (votes.empty())
            for (Track* t : members) {
                const TemplateDecoder& d = t->decoder;
                if (d.last_id != NO_ID && ts - d.last_ok_ts < 1.5 * frame_ms) ++votes[d.last_id];
            }
        int group_id = NO_ID;
        for (auto& [id, n] : votes)
            if (group_id == NO_ID || n > votes[group_id]) group_id = id;
        if (!confirmed && group_id != NO_ID && cfg_.cs_confirm_votes > 0 && votes[group_id] >= cfg_.cs_confirm_votes)
            confirmed = true;
        if (group_id != NO_ID)
            for (Track* t : members)
                if (own(t) == NO_ID &&
                    (t->drone_id != group_id || t->id_confirmed != confirmed || t->inherited())) {
                    t->drone_id = group_id;
                    t->id_confirmed = confirmed;
                    t->inherited_ts = NAN;
                    t->synced_with = gid;
                }
        Constellation c{gid, {}, 0, 0, since, group_id, confirmed};
        for (Track* t : members) {
            c.track_ids.push_back(t->tid);
            c.x += t->x;
            c.y += t->y;
        }
        c.x /= members.size();
        c.y /= members.size();
        constellations_.push_back(std::move(c));
    }
}

// Without constellations: a track blinking exactly like a confirmed one takes its ID.
void Receiver::adopt_synced_ids(double ts) {
    size_t n = static_cast<size_t>(0.6 * CHIPS * cfg_.chip_ms() / std::max(frame_dt_, 1.0));
    if (n < 2) return;
    std::vector<const Track*> refs;
    for (auto& k : tracks_)
        if (k->id_confirmed && k->levels.size() >= n) refs.push_back(k.get());
    if (refs.empty()) return;

    auto tail_stats = [n](const Track& t, std::vector<double>& out, double& mean, double& sd) {
        out.assign(t.levels.end() - n, t.levels.end());
        mean = 0;
        for (double v : out) mean += v;
        mean /= n;
        double ss = 0;
        for (double v : out) ss += (v - mean) * (v - mean);
        sd = std::sqrt(ss / n);
    };
    std::vector<double> a, b;
    for (auto& tp : tracks_) {
        Track& t = *tp;
        if (t.drone_id != NO_ID || t.levels.size() < n || t.contrast < cfg_.min_contrast) continue;
        double ma, sa;
        tail_stats(t, a, ma, sa);
        for (const Track* k : refs) {
            if (std::hypot(k->x - t.x, k->y - t.y) > cfg_.group_radius_px) continue;
            double mb, sb;
            tail_stats(*k, b, mb, sb);
            if (sa < 1e-3 || sb < 1e-3) continue;
            double cov = 0;
            for (size_t i = 0; i < n; ++i) cov += (a[i] - ma) * (b[i] - mb);
            if (cov / n / (sa * sb) >= cfg_.sync_corr) {
                t.drone_id = k->drone_id;
                t.id_confirmed = true;
                t.synced_with = k->tid;
                t.last_seen_ts = ts;
                break;
            }
        }
    }
}

bool Receiver::still_blinking(const Track& t, double ts) const {
    size_t n = static_cast<size_t>(1000.0 / std::max(frame_dt_, 1.0));
    if (ts - t.created_ts < 1000.0 || t.levels.size() < n) return true;
    auto [lo, hi] = std::minmax_element(t.levels.end() - n, t.levels.end());
    return *hi - *lo >= cfg_.min_contrast;
}

void Receiver::drop_lost(double ts) {
    std::vector<std::unique_ptr<Track>> alive;
    for (auto& t : tracks_) {
        bool inside = t->x >= 0 && t->x < w_ && t->y >= 0 && t->y < h_;
        if (ts - t->last_seen_ts < cfg_.lost_ms && inside && still_blinking(*t, ts))
            alive.push_back(std::move(t));
        else if (t->drone_id != NO_ID && t->id_confirmed)
            lost_.push_back({ts, t->x, t->y, t->drone_id});
    }
    tracks_.swap(alive);

    if (!cfg_.constellations) {
        std::set<int> live_tids;
        for (auto& t : tracks_) live_tids.insert(t->tid);
        for (auto& t : tracks_) {
            // `not decoder.confirmed_id` in the reference implementation is also true for ID 0.
            bool unconfirmed = t->decoder.confirmed_id == NO_ID || t->decoder.confirmed_id == 0;
            if (t->synced_with && !live_tids.count(t->synced_with) && unconfirmed) {
                t->synced_with = 0;
                t->id_confirmed = false;
                t->inherited_ts = ts;
            }
        }
    }
    lost_.erase(std::remove_if(lost_.begin(), lost_.end(), [ts](const Lost& l) { return ts - l.ts >= LOST_KEEP_MS; }),
                lost_.end());
}

// Two tracks on one LED: keep the one further along with its read, not just the older one.
void Receiver::dedupe() {
    std::vector<std::unique_ptr<Track>> order;
    for (auto& t : tracks_) order.push_back(std::move(t));
    std::stable_sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
        auto key = [](const Track& t) {
            return std::make_tuple(!t.id_confirmed, t.drone_id == NO_ID, -t.decoder.frames_ok, -t.contrast,
                                   t.created_ts);
        };
        return key(*a) < key(*b);
    });
    tracks_.clear();
    for (auto& t : order) {
        bool separate = std::all_of(tracks_.begin(), tracks_.end(), [&](const auto& k) {
            return std::hypot(t->x - k->x, t->y - k->y) > std::max(6.0, k->radius);
        });
        if (separate) tracks_.push_back(std::move(t));
    }
}

std::vector<Drone> Receiver::group(double ts) {
    const double fx = (w_ / 2.0) / std::tan(cfg_.hfov_deg * M_PI / 180.0 / 2);
    std::map<int, std::vector<const Track*>> by_id;
    for (auto& t : tracks_)
        if (t->drone_id != NO_ID && !t->inherited()) by_id[t->drone_id].push_back(t.get());

    std::vector<Drone> drones;
    for (auto& [id, members] : by_id) {
        const Track* anchor = members.front();
        for (const Track* t : members)
            if (t->core > anchor->core) anchor = t;
        std::vector<const Track*> grp;
        for (const Track* t : members)
            if (std::hypot(t->x - anchor->x, t->y - anchor->y) < cfg_.group_radius_px && t->core >= anchor->core - 3)
                grp.push_back(t);
        if (cfg_.core_drop) {
            auto it = src_core_.find(id);
            auto [prev, prev_ts] = it != src_core_.end() ? it->second : std::make_pair(0.0, ts);
            double src = std::max(anchor->core, prev - 0.01 * (ts - prev_ts));
            src_core_[id] = {src, ts};
            if (anchor->core < src - cfg_.core_drop) continue;
        }
        Drone d{id, 0, 0, 0, 0, static_cast<int>(grp.size()), 0, {}, false};
        double xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY;
        for (const Track* t : grp) {
            d.x += t->x;
            d.y += t->y;
            xmin = std::min(xmin, t->x);
            xmax = std::max(xmax, t->x);
            ymin = std::min(ymin, t->y);
            ymax = std::max(ymax, t->y);
            d.track_ids.push_back(t->tid);
            d.confirmed |= t->id_confirmed;
        }
        d.x /= grp.size();
        d.y /= grp.size();
        d.azimuth_deg = std::atan((d.x - w_ / 2.0) / fx) * 180.0 / M_PI;
        d.elevation_deg = std::atan((h_ / 2.0 - d.y) / fx) * 180.0 / M_PI;
        d.spread_px = std::hypot(xmax - xmin, ymax - ymin);
        drones.push_back(std::move(d));
    }
    return drones;
}

std::vector<Drone> Receiver::process(const uint8_t* bgr, int width, int height, size_t stride, double ts,
                                     const double* cam_rot) {
    if (width != w_ || height != h_) {
        w_ = width;
        h_ = height;
        size_t n = static_cast<size_t>(w_) * h_;
        feat_.assign(n, 0);
        prev_feat_.assign(n, 0);
        diff_.assign(w_, 0);
        hblur_.assign(static_cast<size_t>(HBLUR_ROWS) * w_, 0);
        fdiff_.assign(w_, 0.0f);
        fhblur_.assign(static_cast<size_t>(HBLUR_ROWS) * w_, 0.0f);
        energy_.assign(n, 0.0f);
        have_feat_ = have_energy_ = have_warped_ = false;
        phase_.reset();
        tracks_.clear();
    }
    double dt = std::isnan(last_ts_) ? 0.0 : ts - last_ts_;
    last_ts_ = ts;
    if (dt > 0) frame_dt_ = 0.9 * frame_dt_ + 0.1 * dt;

    if (cam_rot)
        compensate_rotation(cam_rot);
    else if (cfg_.motion_comp)
        compensate_shift(bgr, stride);
    predict(ts, dt);
    update_feature_and_energy(bgr, stride);
    find_peaks();
    assign_and_spawn(ts);

    for (auto& tp : tracks_) {
        Track& t = *tp;
        // The exact source position, when visible, is a second and sharper Kalman measurement.
        double x0 = t.x, y0 = t.y;
        seek_source(t);
        lock_point(t, ts);
        if (t.x != x0 || t.y != y0) correct(t, t.x, t.y, cfg_.kf_fine_px);
        double level = sample(t);
        push_capped<uint8_t>(t.states, t.has_lohi && level > 0.5 * (t.lo + t.hi), LEVELS_LEN);
        t.cs_state = 0;
        if (t.has_lohi && t.contrast >= cfg_.min_contrast) {
            double rel = (level - 0.5 * (t.lo + t.hi)) / (0.5 * t.contrast);
            t.cs_state = rel > cfg_.cs_margin ? 1 : rel < -cfg_.cs_margin ? -1 : 0;
        }
        t.decoder.push(ts, level);
        update_id(t, ts);
    }
    if (cfg_.constellations)
        constellate(ts);
    else if (cfg_.sync_adopt)
        adopt_synced_ids(ts);
    drop_lost(ts);
    dedupe();
    if (cfg_.constellations) {
        std::set<int> live;
        for (auto& t : tracks_) live.insert(t->tid);
        std::vector<Constellation> kept;
        for (auto& c : constellations_) {
            std::vector<int> ids;
            for (int id : c.track_ids)
                if (live.count(id)) ids.push_back(id);
            c.track_ids = ids;
            if (c.track_ids.size() >= 2) kept.push_back(std::move(c));
        }
        constellations_.swap(kept);
    }
    return group(ts);
}

}  // namespace ledrx
