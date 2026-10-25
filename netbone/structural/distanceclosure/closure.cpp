// Transitive closure (Rion Brattig Correia) on an undirected distance graph.
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>

#include "netbone/netbone.hpp"

namespace netbone {

Graph distance_closure(const Graph& d, const std::string& kind, const std::string& algorithm,
                       const std::string& weight, bool only_backbone) {
    if (kind != "metric" && kind != "ultrametric")
        throw std::invalid_argument("Metric not found for this algorithm. Try 'metric' or 'ultrametric',");
    if (algorithm == "dense") throw std::logic_error("Needs some fine tunning.");
    if (algorithm != "dijkstra")
        throw std::invalid_argument("Algorithm implementation not supported. Try 'dense', 'dijkstra' or leave blank.");

    Graph g = d;
    auto lengths = all_pairs_dijkstra_path_length(d, weight, kind == "metric" ? Disjunction::Sum : Disjunction::Max);
    std::string kind_distance = kind + "_distance", is_kind = "is_" + kind;
    std::set<std::pair<int, int>> edges_seen;
    for (int u = 0; u < d.number_of_nodes(); ++u)
        for (int v = 0; v < d.number_of_nodes(); ++v) {
            double length = lengths[u][v];
            if (std::isinf(length) || u == v || !edges_seen.insert({u, v}).second) continue;
            int id = g.find_edge(u, v);
            if (id < 0) {
                if (!only_backbone) g.add_edge(u, v, {{weight, INF}, {kind_distance, length}});
            } else {
                g.set(id, kind_distance, length);
                g.set(id, is_kind, length == g.get(id, weight));
            }
        }
    return g;
}

Graph s_values(const Graph& cm, const std::string& weight_distance, const std::string& weight_metric_distance) {
    Graph g = cm;
    for (const auto& e : g.edges()) {
        double dist = g.get(e.id, weight_distance), metric = g.get(e.id, weight_metric_distance);
        if (dist < INF && metric > 0) g.set(e.id, "s-value", dist / metric);
    }
    return g;
}

Graph b_values(const Graph& cm, const std::string& weight_distance, const std::string& weight_metric_distance) {
    Graph g = cm;
    std::vector<double> mean_distance(g.number_of_nodes());
    for (int k = 0; k < g.number_of_nodes(); ++k) {
        double sum = 0;
        int count = 0;
        for (auto [v, id] : g.adj(k))
            if (g.get(id, weight_distance) < INF) sum += g.get(id, weight_distance), ++count;
        mean_distance[k] = count ? sum / count : NaN;
    }
    for (const auto& e : g.edges())
        if (g.get(e.id, weight_distance) == INF) {
            g.set(e.id, "b_ij-value", mean_distance[e.u] / g.get(e.id, weight_metric_distance));
            g.set(e.id, "b_ji-value", mean_distance[e.v] / g.get(e.id, weight_metric_distance));
        }
    return g;
}

}  // namespace netbone
