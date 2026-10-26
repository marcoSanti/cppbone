/// @file
/// @brief Transitive closure of an undirected distance graph, s-values and b-values (Rion Brattig
/// Correia;
///        distanceclosure/closure.py).

#include <cmath>
#include <stdexcept>
#include <string>

#include "cppbone/cppbone.hpp"

namespace cppbone {

Graph distance_closure(const Graph &d, const std::string_view kind,
                       const std::string_view algorithm, const std::string_view weight,
                       const bool only_backbone) {
    if (kind != "metric" && kind != "ultrametric") {
        throw std::invalid_argument(
            "Metric not found for this algorithm. Try 'metric' or 'ultrametric',");
    }
    if (algorithm == "dense") {
        throw std::logic_error("Needs some fine tunning.");
    }
    if (algorithm != "dijkstra") {
        throw std::invalid_argument(
            "Algorithm implementation not supported. Try 'dense', 'dijkstra' or leave blank.");
    }

    Graph g            = d;
    const auto lengths = all_pairs_dijkstra_path_length(
        d, weight, kind == "metric" ? Disjunction::Sum : Disjunction::Max);

    const std::string kind_distance = std::string(kind) + "_distance";
    const std::string is_kind       = "is_" + std::string(kind);

    // Every ordered pair once; the reverse pair finds the edge added for the first one.
    for (const int u : d.nodes()) {
        for (const int v : d.nodes()) {
            const double length = lengths[u][v];
            if (std::isinf(length) || u == v) {
                continue;
            }
            if (const int id = g.find_edge(u, v); id < 0) {
                if (!only_backbone) {
                    g.add_edge(u, v, {{std::string(weight), INF}, {kind_distance, length}});
                }
            } else {
                g.set(id, kind_distance, length);
                g.set(id, is_kind, length == g.get(id, weight));
            }
        }
    }
    return g;
}

Graph s_values(const Graph &cm, const std::string_view weight_distance,
               const std::string_view weight_metric_distance) {
    Graph g = cm;
    for (const auto &e : g.edges()) {
        const double metric = g.get(e.id, weight_metric_distance);
        if (const double dist = g.get(e.id, weight_distance); dist < INF && metric > 0) {
            g.set(e.id, "s-value", dist / metric);
        }
    }
    return g;
}

Graph b_values(const Graph &cm, const std::string_view weight_distance,
               const std::string_view weight_metric_distance) {
    Graph g = cm;
    std::vector<double> mean_distance(g.number_of_nodes());
    for (const int k : g.nodes()) {
        double sum = 0;
        int count  = 0;
        for (const auto [v, id] : g.adj(k)) {
            if (g.get(id, weight_distance) < INF) {
                sum += g.get(id, weight_distance), ++count;
            }
        }
        mean_distance[k] = count ? sum / count : NaN;
    }
    for (const auto &e : g.edges()) {
        if (g.get(e.id, weight_distance) == INF) {
            g.set(e.id, "b_ij-value", mean_distance[e.u] / g.get(e.id, weight_metric_distance));
            g.set(e.id, "b_ji-value", mean_distance[e.v] / g.get(e.id, weight_metric_distance));
        }
    }
    return g;
}

} // namespace cppbone
