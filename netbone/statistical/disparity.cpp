#include <cmath>

#include "netbone/netbone.hpp"

namespace netbone {

Backbone disparity(const Graph& data, const std::string& weight) {
    Graph g = data;
    auto w = g.edge_values("weight", NaN);
    for (int node = 0; node < g.number_of_nodes(); ++node) {
        int k = g.degree(node);
        if (k <= 1) continue;
        double strength = g.degree(node, weight);
        for (auto [v, id] : g.adj(node)) {
            double alpha = std::pow(1 - w[id] / strength, k - 1);
            double cur = g.get(id, "p_value");
            if (std::isnan(cur) || alpha < cur) g.set(id, "p_value", alpha);
        }
    }
    return {g, "Disparity Filter", "p_value", true, {threshold_filter, fraction_filter}, FilterOn::Edges};
}

}  // namespace netbone
