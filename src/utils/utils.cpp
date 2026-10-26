/// @file
/// @brief Utilities (utils/utils.py): components, cumulative distributions, binomial tails, thread count.

#include <algorithm>
#include <cmath>
#include <functional>

#include "detail.hpp"

namespace cppbone {

void set_num_threads(int n) {
#ifdef _OPENMP
    static const int default_threads = omp_get_max_threads();
    omp_set_num_threads(n > 0 ? n : default_threads);
#else
    (void)n;
#endif
}

int num_threads() { return detail::thread_count(); }

std::vector<std::vector<int>> connected_components(const Graph& g) {
    std::vector<std::vector<int>> comps;
    std::vector<bool> seen(g.number_of_nodes(), false);
    for (int s = 0; s < g.number_of_nodes(); ++s) {
        if (seen[s]) continue;
        comps.emplace_back(1, s);
        seen[s] = true;
        for (size_t i = 0; i < comps.back().size(); ++i)
            for (auto [v, id] : g.adj(comps.back()[i]))
                if (!seen[v]) {
                    seen[v] = true;
                    comps.back().push_back(v);
                }
    }
    return comps;
}

Graph lcc(const Graph& g) {
    auto comps = connected_components(g);
    std::vector<bool> keep(g.number_of_nodes(), false);
    if (comps.empty()) return g;
    auto big = std::max_element(comps.begin(), comps.end(), [](auto& a, auto& b) { return a.size() < b.size(); });
    for (int u : *big) keep[u] = true;
    return g.subgraph(keep);
}

std::vector<std::pair<double, double>> cumulative_dist(std::vector<double> values, bool increasing) {
    if (increasing)
        std::sort(values.begin(), values.end(), std::greater<>());
    else
        std::sort(values.begin(), values.end());
    std::vector<std::pair<double, double>> out;
    for (size_t i = 0; i < values.size(); ++i) out.emplace_back(values[i], double(i + 1) / values.size());
    return out;
}

/// @brief Pi, defined here because M_PI is not standard (MSVC lacks it).
constexpr double PI = 3.14159265358979323846;

/// @brief Error of Stirling's approximation, log(n!) - log(sqrt(2 pi n) (n / e)^n).
///
/// Part of Loader's saddle-point binomial density (as in R's dbinom_raw), which stays accurate for n ~ 1e9 where
/// lgamma differences lose about 6 digits.
/// @param n Argument, > 0.
/// @return The error term.
static double stirlerr(double n) {
    const double S0 = 1. / 12, S1 = 1. / 360, S2 = 1. / 1260, S3 = 1. / 1680, S4 = 1. / 1188;
    if (n <= 15) return std::lgamma(n + 1) - (n + 0.5) * std::log(n) + n - 0.5 * std::log(2 * PI);
    double nn = n * n;
    if (n > 500) return (S0 - S1 / nn) / n;
    if (n > 80) return (S0 - (S1 - S2 / nn) / nn) / n;
    if (n > 35) return (S0 - (S1 - (S2 - S3 / nn) / nn) / nn) / n;
    return (S0 - (S1 - (S2 - (S3 - S4 / nn) / nn) / nn) / nn) / n;
}

/// @brief Deviance term x log(x / np) + np - x, computed without cancellation when x is close to np.
/// @param x Count.
/// @param np Expected count.
/// @return The deviance term.
static double bd0(double x, double np) {
    if (std::abs(x - np) < 0.1 * (x + np)) {
        double v = (x - np) / (x + np), s = (x - np) * v, ej = 2 * x * v;
        v *= v;
        for (int j = 1; j < 1000; ++j) {
            ej *= v;
            double s1 = s + ej / (2 * j + 1);
            if (s1 == s) return s1;
            s = s1;
        }
    }
    return x * std::log(x / np) + np - x;
}

/// @brief Binomial probability Gamma(n+1) / (Gamma(x+1) Gamma(n-x+1)) p^x q^(n-x) by Loader's method.
/// @param x Successes, 0 < x < n.
/// @param n Trials.
/// @param p Success probability.
/// @param q 1 - p.
/// @return The probability.
static double dbinom_raw(double x, double n, double p, double q) {
    double lc = stirlerr(n) - stirlerr(x) - stirlerr(n - x) - bd0(x, n * p) - bd0(n - x, n * q);
    double lf = std::log(2 * PI) + std::log(x) + std::log(n - x) - std::log(n);  // not log1p(-x/n): x/n can round near 1
    return std::exp(lc - 0.5 * lf);
}

/// @brief Binomial probability mass P(X = x), X ~ Binomial(n, p).
/// @param x Successes, 0 <= x <= n.
/// @param n Trials.
/// @param p Success probability.
/// @param q 1 - p.
/// @return The probability.
static double dbinom(double x, double n, double p, double q) {
    if (x == 0) return std::exp(n * std::log1p(-p));
    if (x == n) return std::exp(n * std::log(p));
    return dbinom_raw(x, n, p, q);
}

/// @brief Sums the binomial pmf from `start` stepping away from the mean, where terms shrink geometrically.
/// @param start First term.
/// @param step +1 for the upper tail, -1 for the lower tail.
/// @param n Trials.
/// @param p Success probability.
/// @param q 1 - p.
/// @return The tail probability, summed until the terms stop contributing.
static double tail_sum(double start, int step, double n, double p, double q) {
    double sum = 0, term = dbinom(start, n, p, q);
    for (double j = start; j >= 0 && j <= n && term > 0; j += step) {
        sum += term;
        if (term < sum * 1e-17) break;
        term *= step > 0 ? (n - j) / (j + 1) * (p / q) : j / (n - j + 1) * (q / p);
    }
    return sum;
}

/// @brief Tells whether binomial parameters are invalid (scipy returns NaN for them).
/// @param n Trials.
/// @param p Success probability.
/// @return true if n is not a non-negative integer or p is outside [0, 1].
static bool bad_binom(double n, double p) { return !(n >= 0) || std::floor(n) != n || !(p >= 0 && p <= 1); }

/// Each tail is summed directly on its own side of the mean, so small tails keep full relative precision.
double binom_cdf(double k, double n, double p) {
    if (bad_binom(n, p) || std::isnan(k)) return NaN;
    k = std::floor(k);
    if (k < 0) return 0;
    if (k >= n || p == 0) return 1;
    if (p == 1) return 0;
    double q = 1 - p;
    return k < n * p ? tail_sum(k, -1, n, p, q) : 1 - tail_sum(k + 1, +1, n, p, q);
}

double binom_sf(double k, double n, double p) {
    if (bad_binom(n, p) || std::isnan(k)) return NaN;
    k = std::floor(k);
    if (k < 0) return 1;
    if (k >= n || p == 0) return 0;
    if (p == 1) return 1;
    double q = 1 - p;
    return k + 1 > n * p ? tail_sum(k + 1, +1, n, p, q) : 1 - tail_sum(k, -1, n, p, q);
}

}  // namespace cppbone
