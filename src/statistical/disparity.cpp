/// @file
/// @brief Disparity filter (statistical/disparity.py).

#include <algorithm>
#include <cmath>

#include "../../include/cppbone/detail.hpp"

namespace cppbone {

Backbone disparity(const Graph &data, const std::string &weight) {
    Graph g      = data;
    const auto w = g.edge_values("weight", NaN);
    const int n  = g.number_of_nodes();
    std::vector<double> strength(n);
#pragma omp parallel for schedule(dynamic, 256)
    for (int node = 0; node < n; ++node) {
        strength[node] = g.degree(node, weight);
    }
    // Each edge keeps the smaller alpha of its endpoints of degree > 1, visited in node order (NaN
    // is replaced).
    const auto edges = g.edges();
    const int m      = static_cast<int>(edges.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        const auto &[u, v, id] = edges[i];
        int a = std::min(u, v), b = std::max(u, v);
        bool has   = false;
        double cur = NaN;
        for (const int node : {a, b}) {
            if (node == b && a == b) {
                break; // a self-loop is visited once
            }
            const int k = g.degree(node);
            if (k <= 1) {
                continue;
            }
            if (const double alpha = std::pow(1 - w[id] / strength[node], k - 1);
                !has || std::isnan(cur) || alpha < cur) {
                cur = alpha, has = true;
            }
        }
        if (has) {
            g.set(id, "p_value", cur);
        }
    }
    return {
        std::move(g),   "Disparity Filter", "p_value", true, {threshold_filter, fraction_filter},
        FilterOn::Edges};
}

} // namespace cppbone
