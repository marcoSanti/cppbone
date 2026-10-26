#include <algorithm>
#include <cmath>

#include "detail.hpp"

namespace cppbone {

Backbone disparity(const Graph& data, const std::string& weight) {
    Graph g = data;
    auto w = g.edge_values("weight", NaN);
    int n = g.number_of_nodes();
    std::vector<double> strength(n);
#pragma omp parallel for schedule(dynamic, 256)
    for (int node = 0; node < n; ++node) strength[node] = g.degree(node, weight);
    // Each edge keeps the smaller alpha of its endpoints of degree > 1, visited in node order (NaN is replaced).
    auto edges = g.edges();
    int m = (int)edges.size();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        const auto& e = edges[i];
        int a = std::min(e.u, e.v), b = std::max(e.u, e.v);
        bool has = false;
        double cur = NaN;
        for (int node : {a, b}) {
            if (node == b && a == b) break;  // a self-loop is visited once
            int k = g.degree(node);
            if (k <= 1) continue;
            double alpha = std::pow(1 - w[e.id] / strength[node], k - 1);
            if (!has || std::isnan(cur) || alpha < cur) cur = alpha, has = true;
        }
        if (has) g.set(e.id, "p_value", cur);
    }
    return {std::move(g), "Disparity Filter", "p_value", true, {threshold_filter, fraction_filter}, FilterOn::Edges};
}

}  // namespace cppbone
