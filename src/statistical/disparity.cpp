/// @file
/// @brief Disparity filter (statistical/disparity.py).

#include <cmath>

#include "../../include/cppbone/detail.hpp"

namespace cppbone {

Backbone disparity(const Graph &data, const std::string_view weight) {
    Graph g             = data;
    const auto w        = g.edge_values("weight", NaN);
    const auto strength = detail::strengths(g, weight);
    // Each edge keeps the smaller alpha of its endpoints of degree > 1.
    const auto edges    = g.edges();
    const int m         = static_cast<int>(edges.size());
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        const auto &[u, v, id] = edges[i];
        const auto alpha       = detail::endpoint_min(
            u, v, [&](const int node) { return g.degree(node) <= 1; },
            [&](const int node) {
                return std::pow(1 - w[id] / strength[node], g.degree(node) - 1);
            });
        if (alpha) {
            g.set(id, "p_value", *alpha);
        }
    }
    return {.graph         = std::move(g),
            .method_name   = "Disparity Filter",
            .property_name = "p_value",
            .ascending     = true,
            .filters       = {threshold_filter, fraction_filter}};
}

} // namespace cppbone
