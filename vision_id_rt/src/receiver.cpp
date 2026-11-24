#include "receiver.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <tuple>

#include "codebook.h"
#include "kernels.hpp"
#include "stats.hpp"

namespace ledrx {
namespace {

constexpr size_t LEVELS_LEN = 90;
constexpr size_t OBS_LEN = 40;
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

Receiver::Receiver(Config cfg) : cfg_(cfg) {}

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
    const bool exact = gain == 1.0f;

    // One pass over rows: feature and horizontal blur run two rows ahead of the vertical blur.
    auto slot = [&](int y) { return static_cast<size_t>(y % HBLUR_ROWS) * w; };
    auto make_row = [&](int y) {
        uint8_t* cur = feat_.data() + static_cast<size_t>(y) * w;
        const uint8_t* prev = prev_feat_.data() + static_cast<size_t>(y) * w;
        nonblue_row(bgr + y * stride, cur, w);
        if (exact)
            absdiff_hblur_row(cur, prev, diff_.data(), hblur_.data() + slot(y), w);
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
}

// Auto exposure changes the whole image at once; without compensation every edge would blink.
float Receiver::exposure_gain(const uint8_t* bgr, size_t stride) const {
    std::vector<float> ratios;
    for (int y = 0; y < h_; y += 8) {
        const uint8_t* row = bgr + y * stride;
        for (int x = 0; x < w_; x += 8) {
            int b = row[3 * x], g = row[3 * x + 1], r = row[3 * x + 2];
            int c = std::max(std::max(r, g) - b, 0), p = prev_feat_[static_cast<size_t>(y) * w_ + x];
            if (c > 20 && c < 235 && p > 20 && p < 235) ratios.push_back(static_cast<float>(c) / p);
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

void Receiver::predict(double ts, double dt) {
    if (dt <= 0) return;
    for (auto& t : tracks_)
        if (ts - t->last_seen_ts < cfg_.cv_max_gap_ms) {
            t->x += t->vx * dt;
            t->y += t->vy * dt;
        }
}

void Receiver::snap_and_spawn(double ts) {
    std::vector<Track*> order;
    for (auto& t : tracks_) order.push_back(t.get());
    std::stable_sort(order.begin(), order.end(), [](Track* a, Track* b) { return a->created_ts < b->created_ts; });
    std::vector<bool> used(peaks_.size(), false);
    for (Track* t : order) {
        double reach = cfg_.snap_px + t->radius + std::min(cfg_.search_grow * (ts - t->last_seen_ts), 60.0);
        int best = -1;
        for (size_t i = 0; i < peaks_.size(); ++i) {
            if (used[i] || std::hypot(peaks_[i].x - t->x, peaks_[i].y - t->y) > reach) continue;
            if (best < 0 || peaks_[i].energy > peaks_[best].energy) best = static_cast<int>(i);
        }
        if (best < 0) continue;
        used[best] = true;
        const Peak& p = peaks_[best];
        t->x += cfg_.snap_gain * (p.x - t->x);
        t->y += cfg_.snap_gain * (p.y - t->y);
        t->radius = 0.9 * t->radius + 0.1 * std::max(p.radius, 1.0);
        t->last_seen_ts = ts;
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

void Receiver::update_velocity(Track& t, double ts) {
    if (t.last_seen_ts == ts) push_capped(t.obs, {ts, t.x, t.y}, OBS_LEN);
    std::vector<Track::Obs> pts;
    for (const auto& o : t.obs)
        if (ts - o.ts <= cfg_.cv_window_ms) pts.push_back(o);
    if (pts.size() < 3 || pts.back().ts - pts.front().ts < 3 * frame_dt_) {
        t.vx = t.vy = 0.0;
        return;
    }
    double mt = 0, mx = 0, my = 0;
    for (const auto& o : pts) {
        mt += o.ts;
        mx += o.x;
        my += o.y;
    }
    mt /= pts.size();
    mx /= pts.size();
    my /= pts.size();
    double den = 0, nx = 0, ny = 0;
    for (const auto& o : pts) {
        double dt = o.ts - mt;
        den += dt * dt;
        nx += dt * (o.x - mx);
        ny += dt * (o.y - my);
    }
    double vx = nx / den, vy = ny / den, sp = std::hypot(vx, vy);
    if (sp > cfg_.cv_max_px_per_ms) {
        vx *= cfg_.cv_max_px_per_ms / sp;
        vy *= cfg_.cv_max_px_per_ms / sp;
    }
    t.vx = vx;
    t.vy = vy;
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
    if (t.levels.size() >= 6 && (t.levels.size() % 3 == 0 || !t.has_lohi)) {
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
    if (t.synced_with) {
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

// A track blinking exactly like a confirmed one is the same light (LED and its reflection, or
// the LED re-acquired) and takes its ID without waiting for a full read.
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
    auto first = t.levels.end() - n;
    auto [lo, hi] = std::minmax_element(first, t.levels.end());
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

    std::set<int> live_tids;
    for (auto& t : tracks_) live_tids.insert(t->tid);
    for (auto& t : tracks_) {
        // ID 0 counts as unconfirmed here, as in the reference implementation.
        bool unconfirmed = t->decoder.confirmed_id == NO_ID || t->decoder.confirmed_id == 0;
        if (t->synced_with && !live_tids.count(t->synced_with) && unconfirmed) {
            t->synced_with = 0;
            t->id_confirmed = false;
            t->inherited_ts = ts;
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

std::vector<Drone> Receiver::process(const uint8_t* bgr, int width, int height, size_t stride, double ts) {
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
        have_feat_ = have_energy_ = false;
        tracks_.clear();
    }
    update_feature_and_energy(bgr, stride);

    double dt = std::isnan(last_ts_) ? 0.0 : ts - last_ts_;
    last_ts_ = ts;
    if (dt > 0) frame_dt_ = 0.9 * frame_dt_ + 0.1 * dt;

    predict(ts, dt);
    find_peaks();
    snap_and_spawn(ts);
    for (auto& tp : tracks_) {
        Track& t = *tp;
        seek_source(t);
        lock_point(t, ts);
        update_velocity(t, ts);
        double level = sample(t);
        push_capped<uint8_t>(t.states, t.has_lohi && level > 0.5 * (t.lo + t.hi), LEVELS_LEN);
        t.decoder.push(ts, level);
        update_id(t, ts);
    }
    adopt_synced_ids(ts);
    drop_lost(ts);
    dedupe();
    return group(ts);
}

}  // namespace ledrx
