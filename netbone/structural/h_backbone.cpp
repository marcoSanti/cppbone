#include <algorithm>
#include <cmath>

#include "netbone/netbone.hpp"

namespace netbone {

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
    std::vector<double> weight_values, bridge_values;
    for (const auto& e : g.edges()) {
        double bridge = std::round(bc[e.id] / g.number_of_nodes() * 1000) / 1000;
        g.set(e.id, "h_bridge", bridge);
        bridge_values.push_back(bridge);
        weight_values.push_back(g.get(e.id, "weight"));
    }
    double h_weight = h_index(weight_values);
    double h_bridge = h_index(bridge_values);
    for (const auto& e : g.edges())
        g.set(e.id, "in_backbone", g.get(e.id, "h_bridge") >= h_bridge || g.get(e.id, "weight") >= h_weight);
    return {g, "H-Backbone Filter", "h_bridge", false, {boolean_filter}, FilterOn::Edges};
}

}  // namespace netbone
