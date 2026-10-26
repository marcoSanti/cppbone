/// @file
/// @brief Planar Maximally Filtered Graph (structural/pmfg.py), on Boost's Boyer-Myrvold planarity
/// test.

#include <algorithm>
#include <functional>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/boyer_myrvold_planar_test.hpp>

#include "../../include/cppbone/boost_graph.hpp"
#include "cppbone/cppbone.hpp"

namespace cppbone {

bool is_planar(const Graph &g) {
    return boost::boyer_myrvold_planarity_test(detail::to_boost(g, false));
}

Backbone pmfg(const Graph &data) {
    Graph g   = data;
    auto rows = g.edges();
    for (const auto &e : rows) {
        g.set(e.id, "in_backbone", false);
    }
    const auto w = g.edge_values("weight", NaN);
    std::ranges::stable_sort(rows, std::greater<>(), [&](const EdgeRef &e) { return w[e.id]; });
    // The growing backbone, on the input's node indices. Self-loops never affect planarity, so they
    // are only counted, not added to the graph that is tested.
    using Planar = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;
    const int n  = g.number_of_nodes();
    Planar backbone(n);
    int kept = 0;
    for (const auto &[u, v, id] : rows) {
        if (u != v) {
            const auto [e, added] = boost::add_edge(u, v, backbone);
            if (!boost::boyer_myrvold_planarity_test(backbone)) {
                boost::remove_edge(e, backbone);
                continue;
            }
        }
        g.set(id, "in_backbone", true);
        if (++kept == 3 * (n - 2)) {
            break;
        }
    }
    return {.graph         = std::move(g),
            .method_name   = "Planar Maximally Filtered Graph",
            .property_name = "weight",
            .filters       = {boolean_filter}};
}

} // namespace cppbone
