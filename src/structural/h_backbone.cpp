/// @file
/// @brief H-Backbone (structural/h_backbone.py).

#include <algorithm>
#include <cmath>

#include "../../include/cppbone/detail.hpp"

namespace cppbone {

/// @brief h-index of a list: the largest h such that h values are >= h (the Python loop, including
/// its edge cases).
/// @param values Values (copied and sorted).
/// @return The h-index, 0 for an empty list.
static double h_index(std::vector<double> values) {
    std::ranges::sort(values);
    double h = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        h = static_cast<double>(values.size() - i);
        if (h <= values[i]) {
            break;
        }
    }
    return h;
}

Backbone h_backbone(const Graph &data) {
    Graph g          = data;
    const auto bc    = edge_betweenness_centrality(g, false, "weight");
    const auto edges = g.edges();
    const int m      = static_cast<int>(edges.size());
    std::vector<double> weight_values(m), bridge_values(m);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        const int id        = edges[i].id;
        const double bridge = std::round(bc[id] / g.number_of_nodes() * 1000) / 1000;
        g.set(id, "h_bridge", bridge);
        bridge_values[i] = bridge;
        weight_values[i] = g.get(id, "weight");
    }
    const double h_weight = h_index(weight_values);
    const double h_bridge = h_index(bridge_values);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        g.set(edges[i].id, "in_backbone",
              bridge_values[i] >= h_bridge || weight_values[i] >= h_weight);
    }
    return {.graph         = std::move(g),
            .method_name   = "H-Backbone Filter",
            .property_name = "h_bridge",
            .filters       = {boolean_filter}};
}

} // namespace cppbone
