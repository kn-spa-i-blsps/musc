#include "constellation.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

#include "stats.hpp"

namespace ledrx {

BlinkGrouper::BlinkGrouper(double code_ms, double chip_ms, double min_events, double radius, double block_level,
                           double near_weight)
    : radius_px(radius), code_ms_(code_ms), chip_ms_(chip_ms), min_events_(min_events), block_level_(block_level),
      near_weight_(near_weight) {}

void BlinkGrouper::reindex(const std::vector<int>& tids) {
    if (tids == tids_) return;
    std::map<int, int> pos;
    for (int i = 0; i < n_; ++i) pos[tids_[i]] = i;
    const int n = static_cast<int>(tids.size());
    std::vector<int> old_of(n, -1);
    for (int i = 0; i < n; ++i) {
        auto it = pos.find(tids[i]);
        if (it != pos.end()) old_of[i] = it->second;
    }
    std::vector<int8_t> last(n, 0), ev(n, 0);
    std::vector<int> run(static_cast<size_t>(n) * n, 0);
    std::vector<double> sync(run.size(), 0.0), strikes(run.size(), 0.0), ok_ms(run.size(), 0.0);
    for (int i = 0; i < n; ++i) {
        if (old_of[i] < 0) continue;
        last[i] = last_[old_of[i]];
        ev[i] = ev_prev_[old_of[i]];
        for (int j = 0; j < n; ++j) {
            if (old_of[j] < 0) continue;
            size_t a = static_cast<size_t>(i) * n + j, b = static_cast<size_t>(old_of[i]) * n_ + old_of[j];
            run[a] = run_[b];
            sync[a] = sync_[b];
            strikes[a] = strikes_[b];
            ok_ms[a] = ok_ms_[b];
        }
    }
    last_.swap(last);
    ev_prev_.swap(ev);
    run_.swap(run);
    sync_.swap(sync);
    strikes_.swap(strikes);
    ok_ms_.swap(ok_ms);
    tids_ = tids;
    n_ = n;
    std::set<int> live(tids.begin(), tids.end());
    for (auto it = gid_.begin(); it != gid_.end();) it = live.count(it->first) ? std::next(it) : gid_.erase(it);
}

void BlinkGrouper::block(int a, int b) {
    auto ia = std::find(tids_.begin(), tids_.end(), a), ib = std::find(tids_.begin(), tids_.end(), b);
    if (ia == tids_.end() || ib == tids_.end() || a == b) return;
    int i = static_cast<int>(ia - tids_.begin()), j = static_cast<int>(ib - tids_.begin());
    at(strikes_, i, j) = at(strikes_, j, i) = std::max(at(strikes_, i, j), 2.0);
    at(sync_, i, j) = at(sync_, j, i) = 0.0;
}

std::map<int, int> BlinkGrouper::update(double ts, const std::vector<int>& tids, const std::vector<double>& xy,
                                        const std::vector<int>& states) {
    reindex(tids);
    const int n = n_;
    double dt = have_ts_ ? ts - last_ts_ : 0.0;
    have_ts_ = true;
    last_ts_ = ts;
    if (dt > 0) frame_dt_ = 0.9 * frame_dt_ + 0.1 * dt;
    if (n == 0) return {};

    std::vector<int8_t> ev(n);
    for (int i = 0; i < n; ++i) {
        int8_t s = static_cast<int8_t>(states[i]);
        ev[i] = (s != 0 && last_[i] != 0 && s != last_[i]) ? s : 0;
        if (s != 0) last_[i] = s;
    }
    // A dark chip lasts one chip; "against" means disagreeing for half a chip, but at least 2 frames.
    const int need = std::max(2, py_round(0.5 * chip_ms_ / std::max(frame_dt_, 1.0)));
    const double decay = std::pow(0.5, std::max(dt, 0.0) / code_ms_);
    const size_t nn = static_cast<size_t>(n) * n;
    std::vector<char> blocked(nn), edge(nn);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            size_t k = static_cast<size_t>(i) * n + j;
            int prod = states[i] * states[j];
            if (prod < 0) ++run_[k];
            else if (prod > 0) run_[k] = 0;
            bool strike = prod < 0 && run_[k] == need;
            int ei = ev[i], ej = ev[j], pi = ev_prev_[i], pj = ev_prev_[j];
            bool same = ei != 0 && ej == ei;
            bool near = (ei != 0 && ej == 0 && pj == ei) || (ej != 0 && ei == 0 && pi == ej);
            sync_[k] = sync_[k] * decay + (same + near_weight_ * near);
            strikes_[k] = strikes_[k] * decay + 2.0 * strike;
            blocked[k] = strikes_[k] >= block_level_;
            if (blocked[k] || i == j) sync_[k] = 0.0;
            double dist = std::hypot(xy[2 * i] - xy[2 * j], xy[2 * i + 1] - xy[2 * j + 1]);
            bool ok = sync_[k] >= min_events_ - 0.5 && !blocked[k] && dist <= radius_px;
            ok_ms_[k] = ok ? ok_ms_[k] + std::max(dt, 0.0) : 0.0;
            // The match has to hold for a chip: a common dark gap of two different codes usually splits soon.
            edge[k] = ok && ok_ms_[k] >= chip_ms_;
        }
    ev_prev_ = ev;
    return cluster(tids, edge, blocked);
}

std::map<int, int> BlinkGrouper::cluster(const std::vector<int>& tids, const std::vector<char>& edge,
                                         const std::vector<char>& blocked) {
    const int n = n_;
    std::vector<int> comp(n);
    std::iota(comp.begin(), comp.end(), 0);
    std::map<int, std::vector<int>> members;
    for (int i = 0; i < n; ++i) members[i] = {i};
    std::vector<std::pair<int, int>> pairs;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (edge[static_cast<size_t>(i) * n + j]) pairs.emplace_back(i, j);
    std::stable_sort(pairs.begin(), pairs.end(), [&](auto a, auto b) {
        return sync_[static_cast<size_t>(a.first) * n + a.second] > sync_[static_cast<size_t>(b.first) * n + b.second];
    });
    for (auto [i, j] : pairs) {
        int a = comp[i], b = comp[j];
        if (a == b) continue;
        bool conflict = false;
        for (int x : members[a])
            for (int y : members[b]) conflict |= blocked[static_cast<size_t>(x) * n + y] != 0;
        if (conflict) continue;
        for (int x : members[b]) comp[x] = a;
        auto& ma = members[a];
        ma.insert(ma.end(), members[b].begin(), members[b].end());
        members.erase(b);
    }

    // Stable ids: bigger groups pick first and keep the id most of their tracks had.
    std::vector<std::vector<int>> order;
    for (auto& [k, mem] : members) order.push_back(mem);
    auto min_tid = [&](const std::vector<int>& mem) {
        int m = tids[mem[0]];
        for (int i : mem) m = std::min(m, tids[i]);
        return m;
    };
    std::stable_sort(order.begin(), order.end(), [&](const auto& a, const auto& b) {
        if (a.size() != b.size()) return a.size() > b.size();
        return min_tid(a) < min_tid(b);
    });
    std::map<int, int> out;
    std::set<int> taken;
    for (const auto& mem : order) {
        std::map<int, int> votes;
        for (int i : mem) {
            auto it = gid_.find(tids[i]);
            if (it != gid_.end() && !taken.count(it->second)) ++votes[it->second];
        }
        int gid;
        if (!votes.empty()) {
            gid = votes.begin()->first;
            for (auto& [g, c] : votes)
                if (c > votes[gid]) gid = g;
        } else {
            gid = next_gid_++;
        }
        taken.insert(gid);
        for (int i : mem) out[tids[i]] = gid;
    }
    gid_ = out;
    return out;
}

}  // namespace ledrx
