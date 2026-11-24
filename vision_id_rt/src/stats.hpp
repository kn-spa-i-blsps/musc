#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

namespace ledrx {

// Percentile with linear interpolation; `v` must be sorted.
template <class T>
double percentile_sorted(const std::vector<T>& v, double q) {
    double pos = q / 100.0 * (v.size() - 1);
    size_t i = static_cast<size_t>(pos);
    double frac = pos - i;
    if (i + 1 >= v.size()) return v.back();
    return v[i] + frac * (static_cast<double>(v[i + 1]) - v[i]);
}

template <class T>
void percentiles(std::vector<T>& scratch, double q1, double q2, double& p1, double& p2) {
    std::sort(scratch.begin(), scratch.end());
    p1 = percentile_sorted(scratch, q1);
    p2 = percentile_sorted(scratch, q2);
}

template <class T>
double median(std::vector<T>& scratch) {
    size_t n = scratch.size();
    std::nth_element(scratch.begin(), scratch.begin() + n / 2, scratch.end());
    double hi = scratch[n / 2];
    if (n % 2) return hi;
    double lo = *std::max_element(scratch.begin(), scratch.begin() + n / 2);
    return 0.5 * (lo + hi);
}

// Round half to even.
inline int py_round(double x) { return static_cast<int>(std::nearbyint(x)); }

}  // namespace ledrx
