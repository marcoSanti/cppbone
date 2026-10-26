#include "detail.hpp"

namespace cppbone {

Backbone plam(const Graph& data) {
    Graph g = data;
    auto edges = g.edges();
    int m = (int)edges.size(), n = g.number_of_nodes();
    auto w = g.edge_values("weight", NaN);
    // get_max_weight_edge per node: first neighbour with the strictly largest weight
    std::vector<int> max_edge(n, -1);
#pragma omp parallel for schedule(dynamic, 256)
    for (int node = 0; node < n; ++node) {
        double max_weight = -INF;
        for (auto [v, id] : g.adj(node))
            if (w[id] > max_weight) max_weight = w[id], max_edge[node] = id;
    }
    std::vector<char> keep(g.edge_capacity(), 0);
    for (int e : max_edge)
        if (e >= 0) keep[e] = 1;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) g.set(edges[i].id, "in_backbone", keep[edges[i].id] != 0);
    return {std::move(g), "Primary Linkage Analysis", "weight", false, {boolean_filter}, FilterOn::Edges};
}

}  // namespace cppbone
