/// @file
/// @brief Locally Adaptive Network Sparsification (statistical/lans.py).

#include <algorithm>
#include <cmath>
#include <ranges>

#include "../../include/cppbone/detail.hpp"

namespace cppbone {

double compute_pvalue(const Graph &g, int node, double w) {
    const double u_degree = g.degree(node, "weight");
    const double puv      = w / u_degree;
    const auto count      = std::ranges::count_if(
        g.adj(node), [&](const Adjacent &a) { return g.get(a.edge, "weight") / u_degree <= puv; });
    return 1 - static_cast<double>(count) / g.degree(node);
}

Backbone lans(const Graph &data) {
    Graph g             = data;
    const int n         = g.number_of_nodes();
    const auto w        = g.edge_values("weight", NaN);
    // compute_pvalue counts the neighbours with weight / strength <= puv; with each node's ratios
    // sorted that is a binary search, on the same doubles, instead of a scan per edge.
    const auto strength = detail::strengths(g, "weight");
    std::vector<std::vector<double>> ratios(n);
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < n; ++u) {
        // a NaN ratio never compares <=, so it is never counted
        ratios[u] =
            g.adj(u) |
            std::views::transform([&](const Adjacent &a) { return w[a.edge] / strength[u]; }) |
            std::views::filter([](const double r) { return !std::isnan(r); }) |
            std::ranges::to<std::vector>();
        std::ranges::sort(ratios[u]);
    }
    auto pvalue = [&](const int u, const double weight) {
        const double puv = weight / strength[u];
        // ratios <= puv: the ratios before the first one that is not <= puv
        const auto count =
            std::ranges::partition_point(ratios[u], [puv](const double r) { return r <= puv; }) -
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
    return {.graph         = std::move(g),
            .method_name   = "Locally Adaptive Network Sparsification Filter",
            .property_name = "p_value",
            .ascending     = true,
            .filters       = {threshold_filter, fraction_filter}};
}

} // namespace cppbone
