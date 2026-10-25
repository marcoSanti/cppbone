// Backbone subgraph - fast implementation (Rion Brattig Correia, Felipe Xavier Costa).
// Computes the shortest path distance backbone on a weighted graph whose edges are distances.
#include <algorithm>
#include <numeric>

#include "netbone/netbone.hpp"

namespace netbone {

// Removes every edge that is not the shortest path between its endpoints, visiting nodes by decreasing strength.
static Graph compute_backbone(const Graph& d, const std::string& weight, Disjunction disjunction,
                              std::vector<std::pair<std::pair<std::string, std::string>, double>>* s_values) {
    Graph g = d;
    std::vector<int> ordered(g.number_of_nodes());
    std::iota(ordered.begin(), ordered.end(), 0);
    std::vector<double> strength(g.number_of_nodes());
    for (int u = 0; u < g.number_of_nodes(); ++u) strength[u] = g.degree(u, weight);
    std::stable_sort(ordered.begin(), ordered.end(), [&](int a, int b) { return strength[a] > strength[b]; });

    auto w = g.edge_values(weight);
    for (int n : ordered) {
        auto metric_dist = single_source_dijkstra_path_length(g, n, w, disjunction);
        auto neighbors = g.adj(n);  // copy: edges are removed while iterating
        for (auto [v, id] : neighbors)
            if (metric_dist[v] < w[id]) {
                if (s_values) s_values->push_back({{g.label(n), g.label(v)}, w[id] / metric_dist[v]});
                g.remove_edge(id);
            }
    }
    return g;
}

Graph metric_backbone(const Graph& d, const std::string& weight) {
    return compute_backbone(d, weight, Disjunction::Sum, nullptr);
}

Graph ultrametric_backbone(const Graph& d, const std::string& weight) {
    return compute_backbone(d, weight, Disjunction::Max, nullptr);
}

std::pair<Graph, std::vector<std::pair<std::pair<std::string, std::string>, double>>> metric_backbone_distortion(
    const Graph& d, const std::string& weight) {
    std::vector<std::pair<std::pair<std::string, std::string>, double>> s;
    Graph g = compute_backbone(d, weight, Disjunction::Sum, &s);
    return {g, s};
}

}  // namespace netbone
