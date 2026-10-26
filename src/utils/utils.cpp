/// @file
/// @brief Utilities (utils/utils.py): components, cumulative distributions, binomial tails
/// (Boost.Math), thread count.

#include <algorithm>
#include <cmath>
#include <functional>
#include <ranges>

#include <boost/graph/connected_components.hpp>
#include <boost/math/distributions/binomial.hpp>

#include "../../include/cppbone/boost_graph.hpp"
#include "../../include/cppbone/detail.hpp"

namespace cppbone {

void set_num_threads(const int n) {
#ifdef _OPENMP
    static const int default_threads = omp_get_max_threads();
    omp_set_num_threads(n > 0 ? n : default_threads);
#else
    (void) n;
#endif
}

int num_threads() { return detail::thread_count(); }

/// Boost numbers the components in order of their lowest node.
std::vector<std::vector<int>> connected_components(const Graph &g) {
    std::vector<int> component(g.number_of_nodes());
    const int count = boost::connected_components(detail::to_boost(g), component.data());
    std::vector<std::vector<int>> comps(count);
    for (const int u : g.nodes()) {
        comps[component[u]].push_back(u);
    }
    return comps;
}

Graph lcc(const Graph &g) {
    const auto comps = connected_components(g);
    if (comps.empty()) {
        return g;
    }
    std::vector keep(g.number_of_nodes(), false);
    const auto largest =
        std::ranges::max_element(comps, {}, [](const auto &c) { return c.size(); });
    for (const int u : *largest) {
        keep[u] = true;
    }
    return g.subgraph(keep);
}

std::vector<std::pair<double, double>> cumulative_dist(std::vector<double> values,
                                                       bool increasing) {
    if (increasing) {
        std::ranges::sort(values, std::greater<>());
    } else {
        std::ranges::sort(values);
    }
    const auto size = static_cast<double>(values.size());
    return std::views::iota(size_t{0}, values.size()) | std::views::transform([&](const size_t i) {
               return std::pair{values[i], static_cast<double>(i + 1) / size};
           }) |
           std::ranges::to<std::vector>();
}

/// @brief Tells whether binomial parameters are invalid (scipy returns NaN for them).
/// @param n Trials.
/// @param p Success probability.
/// @return true if n is not a non-negative integer or p is outside [0, 1].
static bool bad_binom(const double n, const double p) {
    return !(n >= 0) || std::floor(n) != n || !(p >= 0 && p <= 1);
}

/// Computed with Boost.Math (the same library scipy uses for binom), which keeps full relative
/// precision in the tails.
double binom_cdf(double k, double n, double p) {
    if (bad_binom(n, p) || std::isnan(k)) {
        return NaN;
    }
    k = std::floor(k);
    if (k < 0) {
        return 0;
    }
    if (k >= n) {
        return 1;
    }
    return boost::math::cdf(boost::math::binomial_distribution(n, p), k);
}

/// Computed with Boost.Math (the same library scipy uses for binom), which keeps full relative
/// precision in the tails.
double binom_sf(double k, const double n, const double p) {
    if (bad_binom(n, p) || std::isnan(k)) {
        return NaN;
    }
    k = std::floor(k);
    if (k < 0) {
        return 1;
    }
    if (k >= n) {
        return 0;
    }
    return boost::math::cdf(boost::math::complement(boost::math::binomial_distribution(n, p), k));
}

} // namespace cppbone
