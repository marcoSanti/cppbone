/// @file
/// @brief Backbone subgraph, fast implementation (Rion Brattig Correia, Felipe Xavier Costa; distanceclosure/backbone.py).
///
/// Computes the shortest-path distance backbone of a weighted graph whose edges are distances.
#include <algorithm>
#include <numeric>

#include "boost_graph.hpp"
#include "detail.hpp"

namespace cppbone {

/// @brief Removes every edge that is not the shortest path between its endpoints, visiting nodes by decreasing
///        strength.
///
/// Removing an edge with a strictly shorter alternative changes no shortest-path distance, so a node's search gives
/// the same distances whichever of the earlier removals have been applied. Nodes are therefore processed in
/// batches: the searches of a batch run in parallel on the graph left by the previous batches, each stopping at the
/// node's largest incident weight (nothing farther can beat an edge), and then the batch's removals are replayed in
/// order. One thread reproduces the sequential algorithm; the result and the s-values do not depend on the thread
/// count.
/// @param d Graph whose `weight` attribute holds distances.
/// @param weight Distance attribute.
/// @param disjunction Sum for the metric backbone, Max for the ultrametric one.
/// @param s_values If given, receives the endpoint labels and s-value (distance / shortest path) of every removed
///        edge, in removal order.
/// @return Copy of d without the redundant edges.
static Graph compute_backbone(const Graph& d, const std::string& weight, Disjunction disjunction,
                              std::vector<std::pair<std::pair<std::string, std::string>, double>>* s_values) {
    Graph g = d;
    int n = g.number_of_nodes();
    std::vector<int> ordered(n);
    std::iota(ordered.begin(), ordered.end(), 0);
    std::vector<double> strength(n);
#pragma omp parallel for schedule(static)
    for (int u = 0; u < n; ++u) strength[u] = g.degree(u, weight);
    std::stable_sort(ordered.begin(), ordered.end(), [&](int a, int b) { return strength[a] > strength[b]; });

    auto w = g.edge_values(weight);
    auto b = detail::to_boost(g);
    std::vector<char> removed(g.edge_capacity(), 0);
    // Per batch position: the neighbour slots (index into adj) whose edge has a strictly shorter path, and the
    // path lengths for the s-values.
    const int batch = 16 * detail::thread_count();
    std::vector<std::vector<std::pair<int, double>>> shorter(batch);
    detail::Errors errors;
#pragma omp parallel
    {
        detail::ShortestPaths sp;
        for (int b0 = 0; b0 < n; b0 += batch) {
            int b1 = std::min(n, b0 + batch);
#pragma omp for schedule(dynamic, 1)
            for (int i = b0; i < b1; ++i)
                errors.run([&] {
                    int src = ordered[i];
                    auto& found = shorter[i - b0];
                    found.clear();
                    double cutoff = -INF;
                    for (auto [v, id] : g.adj(src))
                        if (!removed[id]) cutoff = std::max(cutoff, w[id]);
                    if (cutoff == -INF) return;
                    detail::dijkstra(b, src, w, disjunction, sp, cutoff, &removed);
                    const auto& adj = g.adj(src);
                    for (int k = 0; k < (int)adj.size(); ++k)
                        if (!removed[adj[k].second] && sp.dist[adj[k].first] < w[adj[k].second])
                            found.emplace_back(k, sp.dist[adj[k].first]);
                });
            // implicit barrier; one thread replays the batch in order: an edge leaves at its first endpoint
#pragma omp single
            for (int i = b0; i < b1; ++i) {
                int src = ordered[i];
                for (auto [k, dist] : shorter[i - b0]) {
                    auto [v, id] = g.adj(src)[k];
                    if (removed[id]) continue;  // removed earlier in this batch by its other endpoint
                    if (s_values) s_values->push_back({{g.label(src), g.label(v)}, w[id] / dist});
                    removed[id] = 1;
                }
            }
        }
    }
    errors.rethrow();
    g.remove_edges(removed);
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
    return {std::move(g), s};
}

}  // namespace cppbone
