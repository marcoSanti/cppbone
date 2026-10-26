/// @file
/// @brief High Salience Skeleton (structural/high_salience_skeleton.py).

#include "boost_graph.hpp"
#include "detail.hpp"

namespace cppbone {

Backbone high_salience_skeleton(const Graph& data) {
    Graph g = data;
    auto edges = g.edges();
    int m = (int)edges.size();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) g.set(edges[i].id, "distance", 1 / g.get(edges[i].id, "weight"));
    auto distance = g.edge_values("distance");
    auto b = detail::to_boost(g);
    int n = g.number_of_nodes();
    // Integer counts, so the parallel sum is exact.
    std::vector<long> salience(g.edge_capacity(), 0);
    detail::Errors errors;
#pragma omp parallel
    {
        detail::ShortestPaths sp;
#pragma omp for schedule(dynamic, 4)
        for (int source = 0; source < n; ++source)
            errors.run([&] {
                detail::dijkstra(b, source, distance, Disjunction::Sum, sp);
                // each shortest-path-tree edge counts once per source
                for (int u : sp.settled)
                    if (sp.parent_edge[u] >= 0) {
                        long& s = salience[sp.parent_edge[u]];
#pragma omp atomic
                        s += 1;
                    }
            });
    }
    errors.rethrow();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        int id = edges[i].id;
        g.set(id, "salience", double(salience[id]) / n);
        g.set(id, "in_backbone", double(salience[id]) / n > 0.8);
    }
    return {std::move(g), "High Salience Skeleton Filter", "salience", false, {boolean_filter, threshold_filter, fraction_filter},
            FilterOn::Edges};
}

}  // namespace cppbone
