#include "netbone/netbone.hpp"

namespace netbone {

Backbone high_salience_skeleton(const Graph& data) {
    Graph g = data;
    for (const auto& e : g.edges()) g.set(e.id, "distance", 1 / g.get(e.id, "weight"));
    auto distance = g.edge_values("distance");
    std::vector<double> salience(g.edge_capacity(), 0);
    std::vector<int> parent;
    for (int source = 0; source < g.number_of_nodes(); ++source) {
        single_source_dijkstra_path_length(g, source, distance, Disjunction::Sum, &parent);
        // each shortest-path-tree edge counts once per source
        for (int u = 0; u < g.number_of_nodes(); ++u)
            if (parent[u] >= 0) salience[g.find_edge(parent[u], u)] += 1;
    }
    int n = g.number_of_nodes();
    for (const auto& e : g.edges()) {
        g.set(e.id, "salience", salience[e.id] / n);
        g.set(e.id, "in_backbone", salience[e.id] / n > 0.8);
    }
    return {g, "High Salience Skeleton Filter", "salience", false, {boolean_filter, threshold_filter, fraction_filter},
            FilterOn::Edges};
}

}  // namespace netbone
