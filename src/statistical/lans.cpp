/// @file
/// @brief Locally Adaptive Network Sparsification (statistical/lans.py).

#include <algorithm>
#include <cmath>

#include "../../include/cppbone/detail.hpp"

namespace cppbone {

double compute_pvalue(const Graph &g, int node, double w) {
    const double u_degree = g.degree(node, "weight");
    const double puv      = w / u_degree;
    int count             = 0;
    for (auto [v, id] : g.adj(node)) {
        if (g.get(id, "weight") / u_degree <= puv) {
            ++count;
        }
    }
    return 1 - static_cast<double>(count) / g.degree(node);
}

Backbone lans(const Graph &data) {
    Graph g      = data;
    const int n  = g.number_of_nodes();
    const auto w = g.edge_values("weight", NaN);
    // compute_pvalue counts the neighbours with weight / strength <= puv; with each node's ratios
    // sorted that is a binary search, on the same doubles, instead of a scan per edge.
    std::vector<double> strength(n);
    std::vector<std::vector<double>> ratios(n);
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < n; ++u) {
        strength[u] = g.degree(u, "weight");
        for (auto [v, id] : g.adj(u)) {
            if (double r = w[id] / strength[u]; !std::isnan(r)) {
                ratios[u].push_back(r); // a NaN ratio never compares <=, so it is never counted
            }
        }
        std::sort(ratios[u].begin(), ratios[u].end());
    }
    auto pvalue = [&](const int u, const double weight) {
        const double puv = weight / strength[u];
        const auto count =
            std::upper_bound(ratios[u].begin(), ratios[u].end(), puv,
                             [](const double x, const double r) { return !(r <= x); }) -
            ratios[u].begin();
        return 1 - static_cast<double>(count) / g.degree(u);
    };
    const auto edges = g.edges();
    const int m      = static_cast<int>(edges.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        const auto &[u, v, id] = edges[i];
        g.set(id, "p_value", std::min(pvalue(v, w[id]), pvalue(u, w[id])));
    }
    return {std::move(g),
            "Locally Adaptive Network Sparsification Filter",
            "p_value",
            true,
            {threshold_filter, fraction_filter},
            FilterOn::Edges};
}

} // namespace cppbone
