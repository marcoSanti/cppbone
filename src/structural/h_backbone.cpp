#include <algorithm>
#include <cmath>

#include "detail.hpp"

namespace cppbone {

// h-index of a list: the largest h such that h values are >= h (the Python loop, including its edge cases).
static double h_index(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    double h = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        h = double(values.size() - i);
        if (h <= values[i]) break;
    }
    return h;
}

Backbone h_backbone(const Graph& data) {
    Graph g = data;
    auto bc = edge_betweenness_centrality(g, false, "weight");
    auto edges = g.edges();
    int m = (int)edges.size();
    std::vector<double> weight_values(m), bridge_values(m);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        int id = edges[i].id;
        double bridge = std::round(bc[id] / g.number_of_nodes() * 1000) / 1000;
        g.set(id, "h_bridge", bridge);
        bridge_values[i] = bridge;
        weight_values[i] = g.get(id, "weight");
    }
    double h_weight = h_index(weight_values);
    double h_bridge = h_index(bridge_values);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i)
        g.set(edges[i].id, "in_backbone", bridge_values[i] >= h_bridge || weight_values[i] >= h_weight);
    return {std::move(g), "H-Backbone Filter", "h_bridge", false, {boolean_filter}, FilterOn::Edges};
}

}  // namespace cppbone
